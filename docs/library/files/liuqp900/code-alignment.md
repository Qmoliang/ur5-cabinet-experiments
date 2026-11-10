# 主张与冻结代码映射

证明状态和限制与方法初稿一致。源码按原字节公开，哈希可核验。

| ID | 主张 | 类型 | 边界 |
| --- | --- | --- | --- |
| M1_M2 | 实际状态、原行重建与相同状态下的名义速度QP | `implementation` | 矩阵随当前状态改变；原行数值可接受不等于零容差实数可行。 |
| M3_M4_P2 | 初始秩感知进入方向及两关节特征 | `algebraic_conditional` | 只对实际执行秩感知选择的初始端口；后续终态重选保留P而不重做该过程。 |
| INIT | 有限静态终态采样与局部回退认证 | `implementation` | 不是IK穷举或初态到终态连通保证；初始化全成本纳入预算。 |
| ACTIVATION | 原目标停滞触发和扩展激活 | `implementation` | 初始激活e>2mm，扩展激活e>1mm；反馈钟包括零动作/制动，仅排除compute_wait。 |
| M5_M6 | 固定终态线、几何见证和目标侧支持平面 | `implementation` | 实际检查当前q到qg整段；不是只检查终态邻域，也不是优化未来构型。 |
| M7_P8 | 最弱预测余量LP与后验进展检查 | `prediction_only` | 预测改善不保证实际非线性余量单调；回退分支不提供同一改善保证。 |
| M8 | 扩展前保留旧见证任务及全关节零空间准备 | `implementation` | 显式Legacy调用绕过上层posture_policy覆盖；不能仅看最高层__call__概括。 |
| M9 | 扩展后CERT_PREP改为原进入混合场加两维姿态项 | `implementation` | 不是对所有见证行做姿态零空间投影；maximin行仍可添加。 |
| M10_P7 | CERT_ENTER使用完整关节终态收缩提案 | `proposal_level_conditional` | 射线单步性质不覆盖恒速多前缀或按分量制动；没有正速度下界或预算内到达证明。 |
| M11 | 近表面双分量任务与PN姿态准备 | `implementation` | 零空间维数为6-rank(K)，秩退化时不是固定最多四维。 |
| M11b | 离开近区后部位位置任务及PN、JN辅助任务 | `implementation` | 不把已离开近区的部位继续当作恒速切向任务。 |
| BINDING | 当前有效任务的对偶行选择与执行后的任务交接 | `implementation` | 不使用无效求解对偶；首次选择的影子分支与持续阶段影子QP路径略有不同。 |
| M12_P6 | 切向投影非反向性质 | `algebraic_conditional` | 投影非退化且未主动反向时成立；不等于拓扑绕数保持。 |
| RECURRENCE | 构型重复证据触发方向重选 | `heuristic_empirical` | 历史点不是航点；每个任务重置反向额度，不能推出全局有限切换。 |
| M13_M14_P4 | 新Cartesian记忆及原任务恢复门 | `algebraic_conditional` | 新提交门替代旧surface_pending提交门；精确目标纤维保留不自动覆盖1mm容差球。 |
| REJOIN | 原任务恢复不等于恢复无记忆A组 | `implementation` | 附加记忆仍存在；worker直接install绕过_register，不能借其端点下降筛选宣称所有实际步距离下降。 |
| M15_M16_P1 | KKT梯度重写及认证缩放后的重新编码 | `algebraic_conditional` | 依赖可行KKT元组与H0正定；是事后重写，不证明目标推进或新颖性。 |
| NUMERICAL | 辅助解经KKT核验后复用 | `implementation` | pilot_use_certified_auxiliary=True；不是每次独立优化重解。 |
| M17_M18_P3 | 原行可接受的确定性制动续轨迹 | `contract_conditional` | 依赖模型/认证有效、状态一致和原子提交；已知A异常禁止无条件实现保证。 |
| M19 | 恒速前缀、精确静止等待与反馈钟 | `implementation` | 最终worker将last_delay_ticks固定1且显式认证最多10前缀；不能照搬父类“上次延迟决定全部保持长度”说明。 |
| P5 | 端点一致的记忆曲线下界 | `algebraic_conditional` | 曲率和算术误差保守性为前提；nextafter不是整体IEEE算术形式化证明。 |
| A_BOUNDARY | A组没有终态初始化、目标修正或逃逸记忆 | `implementation` | A仍使用新增执行器；不能作为原执行方式的纯组。 |
| GLOBAL | 一般完备性、全局收敛、未见环境和硬实时 | `unsupported` | 未给出证明，历史冻结模型和有限预算实验也不足以支持。 |
| E_REACH | 本批C/A/B安全到达174/186/294，严格128/186/294，各分母300。 | `empirical` | 历史开发矩阵，不是未见场景或900个独立场景；不构成一般到达证明。 |
| E_COMPONENTS | C→A严格增59退1，A→B增109退1；各自的安全到达与严格验收分开。 | `empirical` | 顺序增量设计，未识别全部交互；同终态信息下的子模块消融仍需另做。 |
| E_MECHANISMS | 新增109例均激活continuation，20例绕行，1例构型重复重选。 | `empirical` | 不能把全部109例归功于绕行或重复检测；触发次数不是机制必要性的证明。 |
| E_QP | C有161条运行出现实际原行违规、7条出现LP确认不可行状态；A/B实际原行违规均0，A末态备份失败1。 | `empirical` | 行违规、可行域为空和数值求解失败是不同概念；只对已记录运行成立。 |
| E_COST | 完整墙钟中位数C15.192s、A41.134s、B144.661s；B初始化完整计入。 | `empirical` | 失败长尾及共同成功子集分别比较；不能宣称各案例都加速或硬实时。 |
| M20_P9 | 零初速度、静止求解、恒速保持与分量制动下的理想0.2rad/s有效速度界 | `algebraic_conditional` | 需要当前调度及有效加速度边界；浮点容差另计，非零初速快照不直接适用；不证明认证与低速各自的独立收益。 |

## M1_M2

实际状态、原行重建与相同状态下的名义速度QP

**边界：** 矩阵随当前状态改变；原行数值可接受不等于零容差实数可行。

- [frozen/pilot_worker_goal_plane_memory.py:32](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/pilot_worker_goal_plane_memory.py#L32) · `d53efb57df5702566e722347a4e75477b51bee5c66f65f4e9d62644940a27662`
- [frozen/original_row_certificate.py:8](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/original_row_certificate.py#L8) · `728252c454db725503494edb2e5d95702525ecbe885c9dfe5667c8cd4bbee21b`
- [frozen/original_row_certificate.py:28](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/original_row_certificate.py#L28) · `728252c454db725503494edb2e5d95702525ecbe885c9dfe5667c8cd4bbee21b`

## M3_M4_P2

初始秩感知进入方向及两关节特征

**边界：** 只对实际执行秩感知选择的初始端口；后续终态重选保留P而不重做该过程。

- [frozen/rank_aware_ports.py:9](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/rank_aware_ports.py#L9) · `b7006e49f9caee88709663073daf095b22709c0463efeb816f9469697c7a2a07`
- [frozen/rank_aware_ports.py:15](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/rank_aware_ports.py#L15) · `b7006e49f9caee88709663073daf095b22709c0463efeb816f9469697c7a2a07`
- [frozen/rank_aware_ports.py:51](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/rank_aware_ports.py#L51) · `b7006e49f9caee88709663073daf095b22709c0463efeb816f9469697c7a2a07`
- [frozen/goal_entry_policy_v3.py:6](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_entry_policy_v3.py#L6) · `b5efb2ab586ed9d6e694104b271c19589022a6a052e2e991df3a3ef43db29773`
- [frozen/goal_entry_policy_v6.py:9](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_entry_policy_v6.py#L9) · `df51db8bbd72b7a93d8df9812294060cdb98b5f3a5247ce75d8351fb97229c9b`
- 证明：[条件与推导](proofs.md)，P2

## INIT

有限静态终态采样与局部回退认证

**边界：** 不是IK穷举或初态到终态连通保证；初始化全成本纳入预算。

- [frozen/build_goal_ports.py:1](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/build_goal_ports.py#L1) · `e9748f6be7ee32c1f2c99d22cfe58864142f85b032e0e7faf5eb41653d08987c`
- [frozen/build_rank_aware_ports.py:1](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/build_rank_aware_ports.py#L1) · `d0f132aa0954560040b0965ec8e6d05d02a71eab078eeccd5368a8b7a84694fa`

## ACTIVATION

原目标停滞触发和扩展激活

**边界：** 初始激活e>2mm，扩展激活e>1mm；反馈钟包括零动作/制动，仅排除compute_wait。

- [frozen/continuation_policy.py:80](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/continuation_policy.py#L80) · `78aa3b7dd5f91e5eca0db84c14d2cdfa62cb65d61f6a84db235d983de58e0420`
- [frozen/compatibility_clipped_policy.py:12](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/compatibility_clipped_policy.py#L12) · `c0325055cf8613c65caf27e717c7f5b2f69e36b2098b7e63c91769d1e37406a3`
- [frozen/surface_stall_policy.py:7](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/surface_stall_policy.py#L7) · `5dd5b209a692dbdeaa331e4b0e21580ad7eba2df2098cbcde69f74c9c7c9a518`
- [frozen/surface_handoff_policy.py:6](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/surface_handoff_policy.py#L6) · `586cfb32b33f66793f128c4e95c7e017e8b8b29737398304c52c724cc97067ac`

## M5_M6

固定终态线、几何见证和目标侧支持平面

**边界：** 实际检查当前q到qg整段；不是只检查终态邻域，也不是优化未来构型。

- [frozen/diagnostic_exact_pruned_policy.py:24](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/diagnostic_exact_pruned_policy.py#L24) · `afb21da6c0044ab7d19900280cfef68282acf83f650024f4e9f71376659b52f0`
- [frozen/goal_side_plane_policy.py:9](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_side_plane_policy.py#L9) · `d14fa1d912182a6ca0968fa94514488c7ab16125c699f1d648dd1175590a41cb`
- [frozen/goal_side_plane_policy.py:16](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_side_plane_policy.py#L16) · `d14fa1d912182a6ca0968fa94514488c7ab16125c699f1d648dd1175590a41cb`
- [frozen/continuation_certified_policy.py:14](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/continuation_certified_policy.py#L14) · `6508341f197ec56fceaf2b16ec7f5e4d89af170b7dd7ac9b163206e6dcc648e1`

## M7_P8

最弱预测余量LP与后验进展检查

**边界：** 预测改善不保证实际非线性余量单调；回退分支不提供同一改善保证。

- [frozen/continuation_maximin_policy.py:13](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/continuation_maximin_policy.py#L13) · `b03b05e998b6e5cf02dd6b839b61c8dd6c42f41f319af710ff12d44357f5ba56`
- [frozen/continuation_maximin_policy.py:9](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/continuation_maximin_policy.py#L9) · `b03b05e998b6e5cf02dd6b839b61c8dd6c42f41f319af710ff12d44357f5ba56`
- [frozen/goal_plane_acceptance_policy.py:7](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_plane_acceptance_policy.py#L7) · `0d916d6c1a10ed6ef1ae626907e46afb6857596a9e449ca7a9313f82c21aa2a6`
- 证明：[条件与推导](proofs.md)，P8

## M8

扩展前保留旧见证任务及全关节零空间准备

**边界：** 显式Legacy调用绕过上层posture_policy覆盖；不能仅看最高层__call__概括。

- [frozen/compatibility_clipped_policy.py:9](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/compatibility_clipped_policy.py#L9) · `c0325055cf8613c65caf27e717c7f5b2f69e36b2098b7e63c91769d1e37406a3`
- [frozen/continuation_policy.py:61](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/continuation_policy.py#L61) · `78aa3b7dd5f91e5eca0db84c14d2cdfa62cb65d61f6a84db235d983de58e0420`

## M9

扩展后CERT_PREP改为原进入混合场加两维姿态项

**边界：** 不是对所有见证行做姿态零空间投影；maximin行仍可添加。

- [frozen/posture_policy.py:6](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/posture_policy.py#L6) · `11aaae9ba61f023d654919b8361b694822a4d6d1859762e0322f3c5e88948ab5`
- [frozen/goal_entry_policy.py:50](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_entry_policy.py#L50) · `fd6c8038ab1e214fbd2b04b3169572966f914cdc08f6972bd983afdd6c49e401`

## M10_P7

CERT_ENTER使用完整关节终态收缩提案

**边界：** 射线单步性质不覆盖恒速多前缀或按分量制动；没有正速度下界或预算内到达证明。

- [frozen/continuation_certified_policy.py:8](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/continuation_certified_policy.py#L8) · `6508341f197ec56fceaf2b16ec7f5e4d89af170b7dd7ac9b163206e6dcc648e1`
- [frozen/continuation_policy.py:72](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/continuation_policy.py#L72) · `78aa3b7dd5f91e5eca0db84c14d2cdfa62cb65d61f6a84db235d983de58e0420`
- 证明：[条件与推导](proofs.md)，P7

## M11

近表面双分量任务与PN姿态准备

**边界：** 零空间维数为6-rank(K)，秩退化时不是固定最多四维。

- [frozen/surface_escape_policy.py:9](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/surface_escape_policy.py#L9) · `274d204ba08cf4d454ef4965186326cfb101ca3152c4ee9d18e4565229a6e6ed`

## M11b

离开近区后部位位置任务及PN、JN辅助任务

**边界：** 不把已离开近区的部位继续当作恒速切向任务。

- [frozen/task_cone_policy.py:10](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/task_cone_policy.py#L10) · `1d5add827886aa4035d89d4bdff13a839ec83ff8df1b9633ed5b1ebbfaeba6b8`

## BINDING

当前有效任务的对偶行选择与执行后的任务交接

**边界：** 不使用无效求解对偶；首次选择的影子分支与持续阶段影子QP路径略有不同。

- [frozen/task_cone_policy.py:22](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/task_cone_policy.py#L22) · `1d5add827886aa4035d89d4bdff13a839ec83ff8df1b9633ed5b1ebbfaeba6b8`
- [frozen/frame_transfer_policy.py:7](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/frame_transfer_policy.py#L7) · `73f13736697cbab52067b386b5f9307d7bae7ff1fb53822c57cacd200ab8a257`
- [frozen/binding_backup_policy.py:15](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/binding_backup_policy.py#L15) · `1b2e2752129bd492388f341af23fc0dc35b31a9f8628eec897f7d160d5fb3f5b`
- [frozen/executed_frame_policy.py:17](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/executed_frame_policy.py#L17) · `9576893513596678e92d59020384b8b623c7eb1d5075fc268e77b8cb389d794a`
- [frozen/executed_frame_policy.py:34](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/executed_frame_policy.py#L34) · `9576893513596678e92d59020384b8b623c7eb1d5075fc268e77b8cb389d794a`

## M12_P6

切向投影非反向性质

**边界：** 投影非退化且未主动反向时成立；不等于拓扑绕数保持。

- [frozen/goal_entry_policy_v5.py:35](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_entry_policy_v5.py#L35) · `3e46eb6f270adb5c5b507372db16262ba01b866acef19a93f77fe003ed0d8f08`
- [frozen/binding_backup_policy.py:15](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/binding_backup_policy.py#L15) · `1b2e2752129bd492388f341af23fc0dc35b31a9f8628eec897f7d160d5fb3f5b`
- 证明：[条件与推导](proofs.md)，P6

## RECURRENCE

构型重复证据触发方向重选

**边界：** 历史点不是航点；每个任务重置反向额度，不能推出全局有限切换。

- [frozen/configuration_recurrence_policy.py:17](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/configuration_recurrence_policy.py#L17) · `e98278d5aabd1e850b05127b0c1e5e75b83206d9695b8bee36a0b7830cf2d943`
- [frozen/configuration_recurrence_policy.py:13](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/configuration_recurrence_policy.py#L13) · `e98278d5aabd1e850b05127b0c1e5e75b83206d9695b8bee36a0b7830cf2d943`
- [frozen/configuration_recurrence_policy.py:9](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/configuration_recurrence_policy.py#L9) · `e98278d5aabd1e850b05127b0c1e5e75b83206d9695b8bee36a0b7830cf2d943`

## M13_M14_P4

新Cartesian记忆及原任务恢复门

**边界：** 新提交门替代旧surface_pending提交门；精确目标纤维保留不自动覆盖1mm容差球。

- [frozen/goal_fiber_policy.py:12](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_fiber_policy.py#L12) · `4eb2a4b7914812027fa8d103e5d9eb2b3459381a41d4566d0ce9fb95184dec57`
- [frozen/goal_fiber_policy.py:23](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_fiber_policy.py#L23) · `4eb2a4b7914812027fa8d103e5d9eb2b3459381a41d4566d0ce9fb95184dec57`
- [frozen/goal_fiber_policy.py:41](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_fiber_policy.py#L41) · `4eb2a4b7914812027fa8d103e5d9eb2b3459381a41d4566d0ce9fb95184dec57`
- [frozen/goal_fiber_policy.py:9](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_fiber_policy.py#L9) · `4eb2a4b7914812027fa8d103e5d9eb2b3459381a41d4566d0ce9fb95184dec57`
- [frozen/goal_entry_policy_v5.py:74](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_entry_policy_v5.py#L74) · `3e46eb6f270adb5c5b507372db16262ba01b866acef19a93f77fe003ed0d8f08`
- 证明：[条件与推导](proofs.md)，P4

## REJOIN

原任务恢复不等于恢复无记忆A组

**边界：** 附加记忆仍存在；worker直接install绕过_register，不能借其端点下降筛选宣称所有实际步距离下降。

- [frozen/goal_fiber_policy.py:9](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_fiber_policy.py#L9) · `4eb2a4b7914812027fa8d103e5d9eb2b3459381a41d4566d0ce9fb95184dec57`
- [frozen/goal_fiber_policy.py:18](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_fiber_policy.py#L18) · `4eb2a4b7914812027fa8d103e5d9eb2b3459381a41d4566d0ce9fb95184dec57`
- [frozen/pilot_worker_goal_plane_memory.py:29](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/pilot_worker_goal_plane_memory.py#L29) · `d53efb57df5702566e722347a4e75477b51bee5c66f65f4e9d62644940a27662`

## M15_M16_P1

KKT梯度重写及认证缩放后的重新编码

**边界：** 依赖可行KKT元组与H0正定；是事后重写，不证明目标推进或新颖性。

- [frozen/goal_entry_policy_v6.py:39](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_entry_policy_v6.py#L39) · `df51db8bbd72b7a93d8df9812294060cdb98b5f3a5247ce75d8351fb97229c9b`
- [frozen/feasible_backup_policy.py:71](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/feasible_backup_policy.py#L71) · `a92b4117e56696aeea3841b68298aba99da8cbe6e5317c6151eafb5867f5fb66`
- [frozen/goal_entry_policy.py:143](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_entry_policy.py#L143) · `fd6c8038ab1e214fbd2b04b3169572966f914cdc08f6972bd983afdd6c49e401`
- [frozen/certified_auxiliary.py:5](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/certified_auxiliary.py#L5) · `c4ad16cc4220efd775cf5760e3d436a431e4f28032c5e93f644bf7f8292445b0`
- 证明：[条件与推导](proofs.md)，P1

## NUMERICAL

辅助解经KKT核验后复用

**边界：** pilot_use_certified_auxiliary=True；不是每次独立优化重解。

- [dependencies/code/online/protocol_liuqp_controller.py:2278](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/dependencies/code/online/protocol_liuqp_controller.py#L2278) · `b7b14b0fa3ef38852d0509390fb1c1442e39d3a95fe43112acb2705e8c77fbac`
- [frozen/certified_auxiliary.py:5](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/certified_auxiliary.py#L5) · `c4ad16cc4220efd775cf5760e3d436a431e4f28032c5e93f644bf7f8292445b0`

## M17_M18_P3

原行可接受的确定性制动续轨迹

**边界：** 依赖模型/认证有效、状态一致和原子提交；已知A异常禁止无条件实现保证。

- [frozen/feasible_backup_policy.py:35](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/feasible_backup_policy.py#L35) · `a92b4117e56696aeea3841b68298aba99da8cbe6e5317c6151eafb5867f5fb66`
- [frozen/feasible_backup_policy.py:86](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/feasible_backup_policy.py#L86) · `a92b4117e56696aeea3841b68298aba99da8cbe6e5317c6151eafb5867f5fb66`
- [frozen/selected_backup_policy.py:24](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/selected_backup_policy.py#L24) · `15bcc90ac0f6d5c0b2d5cd4fcc467312bfe6666cdb99380bfccb160601485fb5`
- [frozen/latency_backup_policy.py:46](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/latency_backup_policy.py#L46) · `2d42737569ade7f3ca2ad9e7e58b60acb5a7ed87615c4aee2645e69a2a1f9e51`
- 证明：[条件与推导](proofs.md)，P3

## M19

恒速前缀、精确静止等待与反馈钟

**边界：** 最终worker将last_delay_ticks固定1且显式认证最多10前缀；不能照搬父类“上次延迟决定全部保持长度”说明。

- [frozen/pilot_worker_goal_plane_memory.py:1](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/pilot_worker_goal_plane_memory.py#L1) · `d53efb57df5702566e722347a4e75477b51bee5c66f65f4e9d62644940a27662`
- [frozen/latency_backup_policy.py:21](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/latency_backup_policy.py#L21) · `2d42737569ade7f3ca2ad9e7e58b60acb5a7ed87615c4aee2645e69a2a1f9e51`

## P5

端点一致的记忆曲线下界

**边界：** 曲率和算术误差保守性为前提；nextafter不是整体IEEE算术形式化证明。

- [frozen/legacy_goal_plane_memory_policy.py:7](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/legacy_goal_plane_memory_policy.py#L7) · `153c8a245a9cfa11a1404ea43c6ecbcef189af9d1f2ae6a933d63357a5bbd689`
- [frozen/goal_entry_policy.py:116](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/goal_entry_policy.py#L116) · `fd6c8038ab1e214fbd2b04b3169572966f914cdc08f6972bd983afdd6c49e401`
- [study/audit_strict_memory.py:1](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/study/audit_strict_memory.py#L1) · `4f63819de12ae5259acd38a9e76ed1ba292b8541e538390921eeceb18e72b338`
- 证明：[条件与推导](proofs.md)，P5

## A_BOUNDARY

A组没有终态初始化、目标修正或逃逸记忆

**边界：** A仍使用新增执行器；不能作为原执行方式的纯组。

- [frozen/baseline_policy.py:12](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/baseline_policy.py#L12) · `7e1b6996a05badf91948cea2c40982b535ed29912ebf6a9cc84a8cdd20c00fa9`
- [frozen/baseline_policy.py:31](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/baseline_policy.py#L31) · `7e1b6996a05badf91948cea2c40982b535ed29912ebf6a9cc84a8cdd20c00fa9`
- [frozen/baseline_policy.py:37](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/baseline_policy.py#L37) · `7e1b6996a05badf91948cea2c40982b535ed29912ebf6a9cc84a8cdd20c00fa9`
- [frozen/dispatch_policy.py:1](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/dispatch_policy.py#L1) · `9fd0487d63f2e80c63dc1daecfc7be1637adc47152e9b25df47cd97dd9820eff`

## GLOBAL

一般完备性、全局收敛、未见环境和硬实时

**边界：** 未给出证明，历史冻结模型和有限预算实验也不足以支持。


## E_REACH

本批C/A/B安全到达174/186/294，严格128/186/294，各分母300。

**边界：** 历史开发矩阵，不是未见场景或900个独立场景；不构成一般到达证明。

- 实验证据：[三组对照](results.md)

## E_COMPONENTS

C→A严格增59退1，A→B增109退1；各自的安全到达与严格验收分开。

**边界：** 顺序增量设计，未识别全部交互；同终态信息下的子模块消融仍需另做。

- 实验证据：[三组对照](results.md)

## E_MECHANISMS

新增109例均激活continuation，20例绕行，1例构型重复重选。

**边界：** 不能把全部109例归功于绕行或重复检测；触发次数不是机制必要性的证明。

- 实验证据：[三组对照](results.md)

## E_QP

C有161条运行出现实际原行违规、7条出现LP确认不可行状态；A/B实际原行违规均0，A末态备份失败1。

**边界：** 行违规、可行域为空和数值求解失败是不同概念；只对已记录运行成立。

- 实验证据：[三组对照](results.md)

## E_COST

完整墙钟中位数C15.192s、A41.134s、B144.661s；B初始化完整计入。

**边界：** 失败长尾及共同成功子集分别比较；不能宣称各案例都加速或硬实时。

- 实验证据：[三组对照](results.md)

## M20_P9

零初速度、静止求解、恒速保持与分量制动下的理想0.2rad/s有效速度界

**边界：** 需要当前调度及有效加速度边界；浮点容差另计，非零初速快照不直接适用；不证明认证与低速各自的独立收益。

- [dependencies/code/online/joint_braking.py:26](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/dependencies/code/online/joint_braking.py#L26) · `d00888bc6505f54c49e254114d842d0395f8209861173aaff4d4f0be11c7ba0b`
- [frozen/pilot_worker_goal_plane_memory.py:49](https://github.com/Qmoliang/ur5-cabinet-experiments/blob/main/experiment/liuqp900/frozen/pilot_worker_goal_plane_memory.py#L49) · `d53efb57df5702566e722347a4e75477b51bee5c66f65f4e9d62644940a27662`
- 证明：[条件与推导](proofs.md)，P9
