## Task Packet

- Scope: 恢复纯 LiuQP 轨迹并实现 UR5e 球证书/椭球证书对照。
- Files to read: `model.py`, `liuqp_controller.py`, `run_simulation.py`, `test_reproduction.py`, `README.md`, 现有 `results/*/summary.json`。
- Files allowed to edit: 上述源码、测试、README、本计划目录；只新增结果目录，不覆盖旧 `prescribed` 结果。
- Required skills: `paper-orchestration`, `experiment-results-planning`, `verification`。
- Evidence/data inputs: 本地 LiuQP PDF、IRIS PDF、实际 JSON/CSV 日志。
- Required artifacts: 路径来源说明；椭球公式说明；两种控制器；失败场景；原始日志；汇总表；测试。
- Rejection checks: 不把人工路点称为论文路径；不把椭球证书称为原版 IRIS；不把表征假堵塞等同于一般局部极小；不使用 mock 数据冒充结果。
- Validation commands: 单元测试；球/椭球场景批量运行；MuJoCo 接触检查；CSV/JSON 一致性检查；旧结果哈希/路径存在检查。
