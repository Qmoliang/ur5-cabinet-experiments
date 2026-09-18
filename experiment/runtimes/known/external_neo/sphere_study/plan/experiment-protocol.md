# KV-NEO-02: NEO with spherical obstacle envelopes

Protocol written before main runs. Question: can the published NEO core control law reach the fixed cabinet target using the existing full-volume spherical envelopes?

## Fixed cases and scope
- Primary: NEO, sphere obstacles, influence distance 0.30 m (paper value).
- Secondary: NEO, sphere obstacles, influence distance 0.046 m (previous matched setting).
- References: existing LiuQP sphere run and previous NEO ellipsoid runs. Preserve every old source and result byte-for-byte. No selection or tuning after observing outcomes.
- One deterministic initial configuration and target from assets/scene.json. No training/test split, repeated random trials, statistical success-rate or generalization claim.

## Geometry and controller
Same 65 robot certificate spheres. The same 368 solid cabinet sub-boxes, maximum cell span 0.075 m, each enclosed by a sphere of radius ||half extents||. All eight corners enclosed proves full solid coverage. This sphere construction is our task representation, not a construction claimed by the NEO paper.

Use analytic sphere-sphere separation d=||c_obstacle-p_robot||-r_robot-r_obstacle and normal (c_obstacle-p_robot)/||c_obstacle-p_robot||. Include every pair with d < influence; no Liu plane pruning, NEAR penalty or ellipsoid distance solver. Reuse the unchanged NEO QP equations, manipulability gradient, slack weighting and solver from KV-NEO-01. No retreat heuristic, planner or intermediate goal, as in that public core-example adaptation.

Task adaptations remain explicit: position-only task, 3 slack variables, safety 0.006 m rather than paper 0.05 m; gain 2, speed cap 0.18 m/s, joint padding 0.015 rad, workspace constraints. Same q0, goal, velocity bounds, DT=0.02 s, 3000 steps=60 s. No controller tuning. Success requires error <1 mm at 50 consecutive collision-free endpoints; record first entry and confirmation separately. Sphere geometry is compatible with NEO, but this is not a complete official benchmark reproduction.

## Preflight and evidence
Verify sphere arrays against original sphere proxies, full-solid coverage, all-pair query filtering, signed-distance directional derivative versus robot Jacobians. Existing unchanged NEO algebra verification remains applicable. Freeze sources and hash old sources/results before runs. Save every joint state, end-effector state, cycle metrics, QP status/fallback/residual and both successes and failures.

Independently reconstruct all 3001 states, all 65*368 proxy distances per state, joint/velocity/workspace limits and MuJoCo contacts. Check continuous cabinet separation under piecewise linear joint interpolation using exact sphere-box distance and lever-arm interval bounds. This is not continuous self-collision or dynamic tracking certification. Report endpoint proxy gaps even if a run fails. Do not assert success from endpoint position alone.

Runtime is descriptive only, not a paper-level speed comparison. No module-benefit ablations or robustness claims; the representation swap relative to the previous NEO runs is the comparison here. N/A: class imbalance, XAI. Create a replay with the actual sphere overlays and saved trajectory; failed cases must support replay without a success timestamp.
