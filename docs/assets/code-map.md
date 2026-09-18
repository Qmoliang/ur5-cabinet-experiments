# Code and reproduction guide

The repository has three separate responsibilities: produce a control experiment, retain evidence, and display that evidence.

## Control ownership

1. experiment/batch.py schedules the fixed list of 19 configurations.
2. experiment/run_case.py dispatches K/N cases to runtimes/known, H cases to runtimes/historical, and O/A cases to runtimes/online.
3. Each runtime's runner owns simulated time, current joint configuration and observation requests.
4. Online perception captures depth, builds proxies and publishes completed generations. The control loop uses the last published map.
5. The MVT index selects nearby robot-obstacle candidates. It does not fit obstacle proxies or choose robot velocity.
6. Distance kernels compute geometric quantities used to build constraints. The controller assembles and solves a QP for joint velocity.
7. The runner integrates joint velocity with a nominal 20 ms step and stores the next state.

The known runtime starts with a fixed map. Historical and current online runs use distinct perception pipelines. The NEO adaptation changes control constraints/objectives inside the corresponding experimental setup.

## Evidence and website

- evidence/<case>/replay_q.npy: recorded joint configurations.
- evidence/<case>/replay_error_mm.npy: position error at the saved states.
- evidence/<case>/result.json: completion, confirmation time and final error.
- docs/data/experiments.json: compact curves and map publication metadata for all 19 attempts.
- docs/assets/*-real.mp4 and *-cert.mp4: saved-state renderings for nine cases, at 6 times simulation speed.
- docs/app.js: synchronizes videos, updates the time cursor and filters the outcome table. It never runs a robot controller.
- reports/: audit reports and frozen-file hashes.

Large pair-state tables and complete causal proxy histories remain in the original archive. Accordingly, the public videos can be played independently, but regenerating certificate videos requires that archive. tools/export_demo.py accepts its path.

## Preview

From the repository root, run:

    python tools/preview.py

Then open http://localhost:8765.

## Running the controller code

The frozen source is provided for inspection. A portable clean-machine rerun has not been validated: Windows native components, original paths and CPU-affinity settings require environment-specific work. The archived experiment overview describes the original complete workstation directory, not an installation procedure for this compact snapshot.

## Interpretation

NEO is a local position-task adaptation. Online runs have different motion and sensing histories. Current thin O02 cores must not be presented as the thicker historical 5.1 model. Reaching and absence of recorded penetration do not establish complete online sensing or hardware safety.

