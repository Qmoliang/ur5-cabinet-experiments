"""Fixed NEO sphere-envelope experiment, separate from historical runs."""
from pathlib import Path
import sys, json, hashlib, shutil, time, argparse
import numpy as np
import mujoco

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent
sys.path.insert(0, str(PARENT))
from run_comparison import ROOT, original, dump
from neo_controller import NEOController
from known_volume import build_volume_cover, coverage_audit
from robot import (build_model, build_robot_certificate, set_configuration,
                   attachment_position, certificate_world_state, DT)

CASES = {'neo_sphere_paper': .30, 'neo_sphere_matched': .046}

class SphereDistances:
    def __init__(self, cover):
        assert cover.representation == 'sphere'
        self.cover = cover

    def query(self, positions, radii, influence=np.inf):
        delta = self.cover.centers[None, :, :] - positions[:, None, :]
        length = np.linalg.norm(delta, axis=2)
        distance = length - radii[:, None] - self.cover.sphere_radii[None, :]
        ri, oi = np.nonzero(distance <= influence + 1e-10)
        if np.any(length[ri, oi] < 1e-14):
            raise RuntimeError('Coincident sphere centers: separating normal undefined')
        normal = delta[ri, oi] / length[ri, oi, None]
        return ri, oi, distance[ri, oi], normal, 0.

class SphereNEOController(NEOController):
    def __init__(self, model, data, robot, cover, influence):
        # Parent initialization does not query geometry; replace its backend.
        super().__init__(model, data, robot, cover, influence)
        self.geometry = SphereDistances(cover)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def protected_files():
    paths = [ROOT / p for p in original.source_hashes()]
    paths += list((ROOT / 'results').rglob('*'))
    paths += list((PARENT / 'results').rglob('*'))
    paths += [PARENT / 'neo_controller.py', PARENT / 'run_comparison.py',
              PARENT / 'audit.py', PARENT / 'replay.py']
    return {p.relative_to(ROOT).as_posix(): sha(p) for p in paths if p.is_file()}

def preflight():
    scene = original.load_scene()
    cover = build_volume_cover(scene.boxes, 'sphere', .075)
    old = np.load(ROOT / 'results/sphere/proxies.npz')
    for key in ['centers', 'sphere_radii', 'proxy_ids', 'cell_half_extents']:
        assert np.array_equal(old[key], getattr(cover, key)), key
    coverage = coverage_audit(cover)
    assert coverage['complete_solid_cell_volume_covered']
    model = build_model(scene); data = mujoco.MjData(model)
    robot = build_robot_certificate(model); geom = SphereDistances(cover)
    assert len(robot) == 65 and len(cover.centers) == 368
    history = np.load(ROOT / 'results/ellipsoid/q_history.npy')
    max_derivative_error = 0.
    for q in history[[0, 300, 750, 3000]]:
        set_configuration(model, data, q)
        pos, jac, radii = certificate_world_state(model, data, robot)
        ri, oi, distances, normal, _ = geom.query(pos, radii)
        assert len(distances) == 65*368
        for influence in CASES.values():
            a, b, d, n, _ = geom.query(pos, radii, influence)
            selected = distances <= influence + 1e-10
            assert np.array_equal(a, ri[selected]) and np.array_equal(b, oi[selected])
            assert np.array_equal(d, distances[selected])
        direction = np.array([.2, -.3, .1, .15, -.1, .25]); eps=1e-6
        values=[]
        for sign in [1, -1]:
            set_configuration(model, data, q + sign*eps*direction)
            p, _, r = certificate_world_state(model, data, robot)
            values.append(geom.query(p, r)[2])
        derivative=(values[0]-values[1])/(2*eps)
        analytic=-np.einsum('pi,pij,j->p', normal, jac[ri], direction)
        max_derivative_error=max(max_derivative_error,float(np.max(np.abs(derivative-analytic))))
    assert max_derivative_error < 1e-7
    report=dict(passed=True, identical_to_original_sphere_proxies=True,
                coverage=coverage, robot_spheres=len(robot),
                full_pair_filter_verified=True, maximum_distance_derivative_error=max_derivative_error,
                sphere_radius_range_mm=[float(cover.sphere_radii.min()*1000),float(cover.sphere_radii.max()*1000)])
    dump(HERE/'preflight.json', report)
    print(json.dumps(report), flush=True)

def run_case(case):
    folder=HERE/'results'/case; folder.mkdir(parents=True, exist_ok=False)
    scene=original.load_scene(); cover=build_volume_cover(scene.boxes, 'sphere', .075)
    model=build_model(scene); data=mujoco.MjData(model)
    set_configuration(model, data, np.array(scene.q0))
    robot=build_robot_certificate(model); target=np.array(scene.waypoints[-1])
    ctrl=SphereNEOController(model, data, robot, cover, CASES[case])
    qs=[data.qpos[:6].copy()]; ees=[attachment_position(model,data)]; logs=[]
    hold=0; maxhold=0; entry=None; confirmed=None; failure=None; started=time.perf_counter()
    try:
        for i in range(3000):
            t0=time.perf_counter(); u, metrics=ctrl.solve(target)
            set_configuration(model,data,data.qpos[:6].copy()+DT*u)
            ee=attachment_position(model,data); err=float(np.linalg.norm(ee-target))
            contacts=[float(data.contact[k].dist) for k in range(data.ncon)]
            pen=sum(d < -1e-8 for d in contacts)
            hold=hold+1 if err < .001 and pen==0 else 0; maxhold=max(maxhold,hold)
            if hold==50 and confirmed is None:
                confirmed=(i+1)*DT; entry=(i-48)*DT
            logs.append(dict(metrics,cycle=i,time_s=(i+1)*DT,error_m=err,
                             exact_contacts=int(data.ncon),penetrating_contacts=pen,
                             min_contact_m=min(contacts) if contacts else None,
                             control_wall_ms=1000*(time.perf_counter()-t0)))
            qs.append(data.qpos[:6].copy()); ees.append(ee)
            if (i+1)%250==0:
                print(f'{case}: {i+1}/3000 error={err*1000:.6f} mm hold={hold}',flush=True)
    except Exception as ex:
        failure=repr(ex)
    np.save(folder/'q_history.npy',qs); np.save(folder/'ee_history.npy',ees)
    shutil.copyfile(ROOT/'assets/drawer.xml',folder/'scene.xml')
    shutil.copyfile(ROOT/'results/sphere/proxies.npz',folder/'proxies.npz')
    dump(folder/'cycles.json',logs); dump(folder/'coverage.json',coverage_audit(cover))
    wall=[r['control_wall_ms'] for r in logs]
    summary=dict(case=case,method='NEO core position-task adaptation',representation='sphere',
                 cycles=len(logs),requested_cycles=3000,simulation_duration_s=len(logs)*DT,
                 wall_duration_s=time.perf_counter()-started,final_error_mm=float(np.linalg.norm(ees[-1]-target)*1000),
                 minimum_error_mm=min([r['error_m']*1000 for r in logs],default=None),
                 success=maxhold>=50,maximum_success_hold_cycles=maxhold,
                 first_sustained_entry_time_s=entry,confirmed_success_time_s=confirmed,
                 penetrating_cycles=sum(r['penetrating_contacts']>0 for r in logs),
                 fallback_cycles=sum(bool(r['fallback']) for r in logs),
                 maximum_qp_violation=max([r['qp_violation'] for r in logs],default=0),
                 nonsolved_status_cycles=sum('solved' not in r['status'].lower() for r in logs),
                 wall_ms_p50=float(np.percentile(wall,50)) if wall else None,
                 wall_ms_p99=float(np.percentile(wall,99)) if wall else None,
                 robot_certificate_count=len(robot),proxy_count=len(cover.centers),
                 safety_margin_m=.006,influence_surface_m=CASES[case],
                 q0=list(scene.q0),target_m=target.tolist(),failure=failure)
    dump(folder/'summary.json',summary); print(json.dumps(summary),flush=True)
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--preflight',action='store_true'); args=ap.parse_args()
    preflight()
    if args.preflight: return
    output=HERE/'results'; output.mkdir(exist_ok=False)
    protected=protected_files(); dump(output/'protected_hashes.json',protected)
    paths=[HERE/'experiment.py', HERE/'plan/experiment-protocol.md',PARENT/'neo_controller.py']
    paths += [ROOT/p for p in original.source_hashes()]
    frozen={}
    for p in paths:
        key=p.relative_to(ROOT).as_posix(); dest=output/'source'/key
        dest.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(p,dest); frozen[key]=sha(p)
    dump(output/'source_hashes.json',frozen)
    results=[run_case(case) for case in CASES]
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(output/'comparison.json',results)
    if any(r['failure'] for r in results): raise SystemExit(1)

if __name__=='__main__': main()
