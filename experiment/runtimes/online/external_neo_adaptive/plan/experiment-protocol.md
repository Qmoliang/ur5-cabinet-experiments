# ADAPT-NEO-01: v4.3 / v4.4c online controller comparison

Frozen before main runs. Main question: does the NEO core position controller reach the cabinet goal with v4.3 adaptive surface ellipsoids and v4.4c resolution-bounded irredundant spheres?

## Cases and controls
Four NEO cases: sphere / ellipsoid crossed with influence distances 0.30 m (paper parameter, primary) and 0.046 m (previous local comparison, secondary). Two fresh LiuQP reference runs, one per representation, with the same current online pipeline and fixed legacy settings. Historical frozen v4.3/v4.4c results remain contextual references and are never overwritten. One original q0 and goal, no randomized trials or statistical generalization.

Use camera_quarter scene, two moving D405 cameras wrist/forearm, 320x180, stride 1, CenterVox 7.5 mm, uncertainty-union cap 1.05, proxy scale cap 70 mm, separate Q_core/U/scalar residual. Ellipsoid mode matched as v4.3 (spatially adaptive patches, no core reinflation). Sphere mode adaptive_irredundant as v4.4c. Do not silently substitute the later v5 ellipsoid set-cover method. Inherited observed_only policy, real-time pacing, perception process, perception_heavy affinity, 8 native pair threads, dense map delta logs, sweep audit_only, no command repair. 3000 cycles x 20 ms = 60 s; <1 mm for 50 cycles. No stopping on first success.

Online runs acquire their own causal observations. Different controller trajectories and asynchronous publication produce different point clouds; this is an end-to-end system comparison, not a same-pointcloud isolated representation experiment. Baseline code is current shared code with v4.3/v4.4 settings; historical byte-identical reproduction is not presumed. Sensor visibility, late/never evidence and map ages must be reported. No claim of unknown-space safety under observed_only.

## NEO fidelity
Use unchanged NEO objective: 0.5*.01*||u||^2 + 0.5/e*||delta||^2 - grad(m)^T*u, J*u+delta=v, e=max(L1 position error,1e-6); translational manipulability and central gradient step1e-5. Collision n^T*J_point*u <= (d-.006)/(di-.006), xi=1, all pairs within influence, no Liu NEAR penalty, no ordered plane pruning. Same workspace, velocity/joint bounds; NEO joint damper influence .9 rad, stop .015 rad. Task gain2, speedcap .18. Quadprog0.1.13, same-QP SLSQP only on numerical failure, logged. Position-only / 6mm safety adaptation as earlier study; no paper retreat heuristic, posture target, planner, waypoint or random dither.

Sphere distances include effective sphere radius plus scalar residual once. Ellipsoid separation uses support of Q_core + U as a Minkowski sum plus scalar residual; do not drop uncertainty, fuse it into the core, or use the known-volume single-ellipsoid distance shortcut. Support normals solved with existing native safeguarded geometry kernel; QP is NEO. The new controller subclasses the old class only for geometry storage/interface compatibility and overrides solve entirely.

NEO maintains its own conservative MVT with query padding di+5um; never reuse a 46mm index for 300mm influence. Verify each replacement index against brute AABB oracle. Original guard/map MVT remains unchanged. Record controller index builds separately because they happen at publication.

## Evidence and limitations
Before main runs validate distance derivative, all-pair candidate oracle, QP explicit/eliminated equivalence with uncertainty-bearing real frozen snapshots. Smoke runs validate online integration and failure saving; no parameter tuning on outcomes. Preserve smoke errors and corrections. Snapshot source and protect old frozen manifests.

Save full q/ee histories, per-cycle metrics and solver diagnostics, causal proxies/CenterVox/source configurations, map delays, coverage/scale/MVT gates, sweep failures and contacts. Do not infer successful safe control solely from final error. Audit reconstructed endpoint geometry and physical cabinet separation; distinguish proxy sweep and true geometry. Strict perception gate failures remain failures. QP failure or numerical geometry failure aborts and preserves partial trace; do not replace unsafe/infeasible command with an unlogged fallback.

Report success/hold/times/error; final proxy count/radius distribution; QP fallback/failure/constraint residual; controller p50/p99, map latency; contact/sweep/observability. Solver time and motion time are separate. Six single runs do not establish universal shape/controller superiority. No MVT speedup claim without an acceleration ablation, and no causal attribution to individual NEO terms. No dataset split/imbalance/XAI applies. Render saved trajectories for inspection, no synthetic data figures.
