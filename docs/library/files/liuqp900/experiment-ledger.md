# 实验台账：900、1200及历史尝试的正确口径

本台账服务于[论文初稿](paper-draft.md)。所有旧报告按原文归档，解释更正写在本页；不改写旧判定，不把复制的报告、网页播放或数学核验计成新实验。

## 1. 为什么可以说1200，也为什么它不是项目总量

| 批次 | 运行记录 | 通过结果 | 在论文中的角色 |
| --- | ---: | --- | --- |
| C：原目标＋原执行方式 | 300 | 物理174，严格128 | 主对照 |
| A：原目标＋共同执行器 | 300 | 物理186，严格186 | 主对照 |
| B：终态初始化与新控制＋共同执行器 | 300 | 物理294，严格294 | 主对照 |
| 旧在线椭球RRT系统 final_r17 | 300 | 原协议安全到达298 | 历史系统参照，非同协议第四臂 |
| 旧已知场景系统 | 300 | 原协议通过300 | 另一个历史批次 |
| 旧在线球表示配对 | 300 | 原协议安全到达275 | 几何表示实验，椭球基线只读复用 |

因此，选取前四行时是 **900＋300＝1200条运行记录**；主配对仍是300条件×3组。六行计1800条这些已确认批次的记录，另有开发、诊断、消融和重跑，所以全项目历史总量超过1200。这里不声称已完成全项目跨版本、跨归档的穷尽去重。尤其球/椭球报告中的298/300复用了同一个旧椭球批次，不能再额外加300。

旧600条索引明确区分known/online各300；球表示收尾报告另记新增300、基线复用。来源：[旧索引](https://qmoliang.github.io/ur5-cabinet-experiments/closeout/data.json)、[球表示原报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/sphere-r17-report.json)、[新900索引](https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/data.json)。

**论文建议措辞：** “在300个历史输入条件上完成900次三组冻结控制对照，并提供300次旧在线规划系统运行作为历史参照。”不能写“1200个独立场景”“1200次同协议随机试验”，也不能用298→294推断新方法净退化4例。新旧地图使用、规划参与、时间和严格判据的差异见[版本核对](reconciliation.md)。

## 2. 围绕最终控制的开发证据链

下表数量是对应原始结果表的条目数，含各版开发、原失败复现、回归和部分初始化/启动失败，不是每行最终算法的统一成功率。后文的最终验证子集已包含在对应整批条目里，不再相加。各阶段结果可能共享输入或重演同一时序，不能作为独立样本合并。

| ID／研究 | 原结果表条目 | 与最终方案有关的证据 | 必须保留的限制 |
| --- | ---: | --- | --- |
| H01 固定路径执行消融，11-05 | 8（4条件×2） | LiuQP与普通P跟踪均4/4；原系统成功不能独归LiuQP | 已有认证路径、四条件，不能推论整个300例无QP结果 |
| H02 低维绕行试验，11-06 | 33 | 球/椭球都可在同一货架长期近停滞；实际姿态分支已执行 | 最终改进安全到达0/4；不能证明严格孤立局部极小 |
| H03 阻塞释放，11-06 | 38 | 识别前臂附近阻塞，需要基座/肩部协调退让 | 真实停滞恢复1次成功，同版本2次重复失败；单代理打开不等于持续绕行 |
| H04 目标端进入，11-06/07 | 49 | 静态终态、目标侧局部进入与持续几何记忆接入 | 最终固定快照3/3，重置后原初态2/3，坏延迟0/1；5条原安全审计未过 |
| H05 目标端重选向，11-07 | 15 | 固定坏时序可重演，方向族耗尽不等于物理无路 | 5条原始到达仅2条过原完整审计；补充证书支持5条，原标签不改；确定性重复非独立 |
| H06 整段进入证书，11-07 | 19 | 最终8项7到达；最弱余量与平均余量的两条件对照为2/2与1/2 | 8项含确定性重复；原初态仍失败；19项包含2次启动错误，另1计划未启动不计入 |
| H07 实际进展与目标侧平面，11-07/08 | 62 | 去延迟/去缓存仍失败，保留坏时序而改目标侧平面可到达；同阈值曲率证书修复一例回归 | 最新12/12为同场景开发验证；早期失败和安全审计未通过仍在原表 |
| H08 当时方法留出，11-08 | 24（12对） | 基线3/12，新方法9/12 | 3例初始化失败必须计分母；21条有轨迹；输入后来参与修复，不再是最终独立留出 |
| H09 秩感知初始化，11-08/09 | 54 | 旧初始化失败3/3修复但闭环1/3；旧成功21/21逐状态保持；15新对照9/15→13/15 | 初始化成功不等于到达；不能将不同测试集9/12→13/15直接当涨幅 |
| H10 T0延续/部位调整，11-09 | 156 | 见证满秩抹掉姿态项、虚拟见证切换、真实阻塞部位过期与原QP不可行被分开 | 最终原起点11/12、快照5/12、回归34/35，未获稳定替代验收；84项启动前取消不计运行 |
| H11 T0分项因果与接口修复，11-09 | 131 | 严格静止状态与裁剪后认证的接口修复；同坏时序单因素复验支持裁剪修复 | 最终快照3/4、原起点3/4、回归34/35；末态强制接回续跑不替换原失败 |
| H12 持续绕行与退出最终验证，11-09 | 151 | 最终4快照＋4原起点＋35回归共43/43，追加16/16；时钟消融2/4→4/4 | 43项未启动候选另列；35回归来自24个来源，含快照/重演；没有未见泛化保证 |

H10/H11的未获接受候选不能被H12的成功“覆盖”。H12最终版保留净进展退出门；提前退出、缩小进展门和可逆目标试探等并未稳定获益，没有进入正式B组。完整记录的继承关系通过原报告中的版本、配置和来源字段追踪。

H11同四条固定失败时序的比较为旧局部排斥0/4、Cartesian进展3/4、禁部位切换2/4、仅终态证书退出1/4。它支持对应版本的设计选择，不是最终B组在300条件上的组件消融。H12重复检测启用前4615个反馈状态一致，随后重选使一个失败到达；该单例因果证据不能升级成“109个新增都来自重复检测”。

## 3. 其他路线与未采用方案

- **几何表示。** 球表示新增300次，275通过；旧椭球基线298/300只读复用，逐例椭球独有成功23。该实验包含原在线系统与规划，不能解释新C/A/B共同椭球输入下的控制收益。
- **离线区域覆盖与GCS，48项。** 每方法16个配置，均有6个共同目标池为空、未调用规划器；实际进入求路的10条件中RRT10、IRIS-Fast＋GCS1、IRIS-Precise＋GCS0。Fast还有区域尝试上限导致提前停止，不能把14秒预算的潜力视作已穷尽。[原报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/gcs-pilot-report.txt)
- **工程引导对齐的另一48项。** 原报告中实际调用求路的10条件，当前RRT10、两种IRIS集成各0。它描述当时集成、冷启动和预算，不证明IRIS-ZO原方法普遍不如RRT。相关分支仍有实现/文献对齐待审计问题；尚未获准用于对等方法排名的144项材料不在本次发布中补入。[原报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/iris-alignment-report.txt)
- **6D探索树、旧回退航点提案。** 是历史设计分支，不属于最终冻结B控制。没有在本台账中给仅有推导或未启动方案虚构闭环运行数。
- **区域体积、连接器、端点反馈、采样器等专项检查。** 属于更早规划工程诊断，不能与本论文到达分母相加。主稿只引用与控制归因有关的已核对结论；本页不是项目所有文件的全量目录。

## 4. 计数与证据等级规则

1. 一条正式运行由“批次＋版本＋条件＋方法＋实际尝试ID”识别；同一结果的本地副本、ZIP副本、网页载荷和新汇总不是新运行。
2. 初始化失败计入所分配方法的总尝试分母，但不冒称存在运动轨迹。启动前取消作业不计已执行运行；启动后中止前缀保留并标明。
3. 固定延迟重演用于可复现性或单因素因果诊断；相同输入的自然延迟重复用于有限鲁棒性检查。两者都不是新环境样本。
4. 从失败末态重新给预算续跑是诊断；原失败仍失败，不能将续跑成功倒填原300秒结果。
5. 已确认接口根因、单因素干预支持、日志相容假说及尚无定位，按这四种强度分别叙述。正的单步速度、局部证书或目标侧记忆不作为最终到达证明。
6. 新900三组是顺序增量设计。C→A、A→B不能识别第四臂交互，也不能将认证、低速、停走及初始化等组件各自独立归因。

## 5. 原始来源与完整性

本次归档H01—H12的原始报告及结果JSON副本，另附A/B、C、球表示与两份离线求路报告。原文件的全文、失败标签与版本口径保留；如果历史措辞与最终冻结代码冲突，以[方法更正](author-notes.md)和本稿限定为准，不修改旧报告以掩盖差异。

每个来源的路径、字节数、公开文件SHA-256和结果条目数见[来源清单](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/manifest.json)。报告副本不是历史全部轨迹的再次上传；各研究原轨迹与输入归档继续保留。本次没有生成任何新的机械臂实验运行。


| ID | 原报告 | 原结果表 |
| --- | --- | --- |
| H01 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/execution-ablation-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/execution-ablation-results.json) |
| H02 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/lowdim-pilot-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/lowdim-pilot-results.json) |
| H03 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/release-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/release-results.json) |
| H04 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/goal-entry-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/goal-entry-results.json) |
| H05 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/reheading-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/reheading-results.json) |
| H06 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/entry-certificate-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/entry-certificate-results.json) |
| H07 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/progress-acceptance-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/progress-acceptance-results.json) |
| H08 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/holdout-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/holdout-results.json) |
| H09 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/rank-init-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/rank-init-results.json) |
| H10 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/t0-continuation-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/t0-continuation-results.json) |
| H11 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/t0-causal-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/t0-causal-results.json) |
| H12 | [报告](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/continuation-exit-report.txt) | [结果JSON](https://qmoliang.github.io/ur5-cabinet-experiments/library/files/liuqp900/history-evidence/continuation-exit-results.json) |


主900的冻结输入与源记录哈希、原始证据分包及完整回放仍由[实验网站](https://qmoliang.github.io/ur5-cabinet-experiments/liuqp900/)提供。所有表格由冻结索引重算，见[表格附录](experiment-tables.md)；不以新统计覆盖原JSON。
