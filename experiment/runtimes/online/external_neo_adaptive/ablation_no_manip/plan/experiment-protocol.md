# ADAPT-NEO-02: remove the manipulability objective, execute full online trajectories

Written before runs. Question: does the nonzero one-step command observed at the previous stalled posture translate into actual closed-loop reaching?

Three fixed cases: sphere di=0.30, ellipsoid di=0.30, ellipsoid di=0.046. Start each from the original q0, with fresh causal camera maps, for 3000 steps x 20ms. Compare against the corresponding preserved ADAPT-NEO-01 runs. This is not a continuation from the previous final map and not a same-pointcloud comparison. No repeated tuning or successful-run selection.

Only controller change: manipulability linear objective coefficient 1 -> 0. After slack elimination and positive scaling, retain H=J.T J+.01*e*I and replace g=-J.T v-e*grad(m) with g=-J.T v. All hard constraints, slack bounds, safety 6mm, joint limits, workspace, task gain/speed, collision influences and uncertainty geometry stay unchanged. Still compute grad(m) so the old code path and cost are comparable; no speed optimization is claimed. This is an NEO ablation, not the complete original NEO method.

Keep original v4.3 core/U/residual and v4.4c sphere policies, two cameras, 7.5mm CenterVox, 70mm scale cap, perception_heavy affinity, original asynchronous scheduling, audit-only sweep, no retreat/planner/waypoint/dither. Observed-only scope and late/never limitations remain. Success <1mm for 50 cycles; no early stopping. Follow ADAPT-NEO-01 for all omitted fixed settings.

Preflight: at all three preserved final poses and maps, reproduce previously observed no-manipulability single-QP solutions. Ensure H/A/bounds remain identical and only g changes. Preserve source copies and manifests. New wrapper records the matrices and active obstacle geometry if a QP fails, without command repair.

Report every success, failure, QP abort, contact, proxy sweep and causal-perception failure. Run the same independent full-state physical-cabinet audit and sampled causal proxy audit. If failure aborts, preserve partial states and failing QP rather than extrapolating to 60s. Report movement times separately from runtime. Single asynchronous trials do not establish statistics or universal superiority; no MVT, XAI or generalization claim. Replays use actual active proxy snapshots.
