# 最终能力与证据审计

## 已使用能力

- 本地 MuJoCo/OSQP/NumPy/SciPy 环境：运行 UR5e 顺序 QP、精确接触检查、椭球支持函数和扰动实验。
- Matplotlib/Pillow：从真实 CSV/JSON 生成并解码验证 450 dpi PNG/SVG。
- 单元测试：13 项通过，覆盖原 LiuQP、椭球包围、支持角梯度、最近点法向和一般椭球支持法向。

## 未使用或不可用能力

- 未使用网络数据、外部插件、VCC、RRT、A* 或人工路径生成器。
- 本地 `view_image` 因 Windows sandbox helper refresh 错误无法回读图片；改用 Pillow 完整解码、像素尺寸和文件大小检查。最终 PNG/SVG 已生成。
- 当前子目录不是 Git 仓库，因此没有 Git 状态或提交证据；所有结果均以路径、JSON、CSV、测试输出和验证脚本为证。

## 终止检查

- 主球版与完整椭球版均为 `direct`、`final_goal_only`、名义扰动为 0。
- 固定 204 障碍代理时球版失败、完整椭球版成功已由真实轨迹复现。
- 固定层级球模型闭合证书与 12 组联合小扰动审计均存在。
- 969 障碍球的细化基线成功，已用于阻止“球版绝对不可达”的过强结论。
- 历史 `results/shelf_prescribed`、`results/cage_prescribed` 仍存在；负结果目录未删除。
- 技术说明明确限制：本实验解决表示诱发的目标排除/假闭塞，不声称解决一般全局同伦局部极小。
