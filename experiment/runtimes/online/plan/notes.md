# 研究记录

## 2026-09-02

- 后续中文原理文档以 `LiuQP_Mathematical_Derivation_and_Theory_Guide_English_Expanded.docx` 为母版：公式（1）—（20）的运动学、模块点速度、末端反馈、任务堆叠与关节界直接复用；从工作空间/障碍碰撞几何开始说明量纲并推导椭球版本。
- 源母版采用从机器人指向障碍物的法向 \(\tilde s\)。第一章采用的障碍物指向机器人法向 \(n\) 与其满足 \(n=-\tilde s\)；后续所有公式必须先声明方向，禁止混用。
- 单机器人球—单障碍核心椭球使用一维乘子 Newton、必要二分与乘子热启动；一般支持和使用受保护 Riemannian Newton、Armijo 与法向热启动。两者是按代理类型选择的窄阶段分支，不是连续叠加的两套计算。
- 方向不确定性矩阵为零时，支持点公式中相应分量直接省略，禁止数值计算 \(0/0\)。
- 源 DOCX 已通过 Word 公式对象与精确生成源码核对内容；当前机器缺少 LibreOffice，不能声称完成源文档逐页渲染检查。

## 2026-08-30

- IRIS 在本项目中的作用限定为椭球支持函数、真实最近点法向和分离平面推导；不执行 IRIS 构型空间 MVIE/GCS，也不存在“IRIS-inspired 冗余删除”实验组。
- 冗余删除属于 LiuQP ordered erase-remove。椭球推广以支持最小值 `n^T c - sqrt(n^T Q n) - sqrt(n^T U n) - eta` 判断代理是否完全位于已保留平面的障碍侧；CONTACT/RECOVERY pair 必须保留自己的恢复行。
- 在线含方向不确定性的椭球 pair 求解单位球面上的支持和，使用受保护 Riemannian Newton、Armijo 和法向热启动；已知无方向不确定性的真实点到椭球最近点使用一维 multiplier Newton、必要二分和 multiplier 热启动。
- 原始 `full_scan` 没有 action-distance AABB 判据，会改变进入 erase-remove 的代理集合，不能与 MVT 计时声称“只改数据结构”。纯加速参考必须用执行同一 inclusive float32 AABB 判据的 O(N) 暴力 oracle。
- 正式在线命令只来自 LiuQP；连续 proxy sweep、MuJoCo 真值接触、候选和三态地图重放均是运行后只读审计。
- unknown 维护为 free/occupied 稀疏集合的补集并可视化，但正式主分支是 monitor-only，因此不外推为任意未观测空间绝对安全。
- C++ 支撑核曾把“零值但非空的热启动行”误当成已归一化输入，第一次 Newton 可从 \(S^2\) 外启动。v5 已强制所有初值归一化；残差大于 \(10^{-7}\) 时从中心连线重启 64 次，仍不合格则中止，禁止不准确法向进入 QP。


## 2026-09-15

- 第三章代码阅读采用“主流程向下展开、返回主流程继续”的顺序，主阅读入口固定为 `run.py::run` 和 `ProtocolLiuQPController.solve`，不再从 `collision.cpp` 单独向上猜调用语义。
- 当前 known-volume 椭球默认热路径为 `_prefetch_exact_closest_pairs -> closest_prune_pairs_mvt_warm -> mvt_ellipsoid_closest_prune_pairs_warm -> ellipsoid_closest_prune_pairs_warm -> ellipsoid_closest_points_warm`。
- 当前 \(U_j=0,\delta_j=0\)；`geometry.py` 的 Python 最近点用于数学参考/回退，含不确定性的支持和分支不属于当前运行。
- CONTACT 参数在融合 prune 中用于预识别和补回恢复 pair，不替代 KKT 最近点求解。



- 补充审计确认：球/椭球各只有 6 种 AABB 尺寸，368 个代理全部位于 MVT level 4，层计数为 (0,0,0,0,368)，因此当前实验只使用单有效层的空间哈希作用；实际查询半径为机器人球半径加 0.046005 m。
