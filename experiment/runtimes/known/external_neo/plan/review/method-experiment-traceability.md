# Traceability
| Question | Module | Evidence | Allowed claim |
|---|---|---|---|
| NEO reaches same target? | neo_controller.py | 3000-step logs, q history | This fixed task and parameterization only |
| Same task? | run_comparison.py | original hashes and geometry arrays | Same known volume, robot and target |
| Faithful QP core? | verification.py | derivative and explicit slack checks | Position-task reimplementation, not official full benchmark |
| Collision-free? | audit.py | endpoint contacts and interval box certificate | Only stated audited geometry/time interpolation |
