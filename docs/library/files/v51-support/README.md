# 实验5.1：连续占据支持体椭球

**当前交付版已达到用户要求：最终误差 0.030714 mm < 0.18 mm。** 完整3000周期，控制器p99 10.279 ms，0穿透/0扫掠审计失败。可观测性审计仍未通过；不宣称未知空间全局安全认证。

双击 [查看实验5.1.cmd](查看实验5.1.cmd) 打开最新达标回放。青绿色为真正进入控制的连续体积椭球；空格暂停，F看末态，0从头，Y开关体积层，P点云，Q活动QP代理，R机器人证书球。

完整说明：[达标版本验收报告](docs/达标版本验收报告.md)。首版21.4mm的失败结果保留在[首版报告](docs/首版验收报告.md)。

## 运行和核验

```powershell
& D:/anaconda3/envs/simple/python.exe -B D:/MuJoCo/LiuQP_experiment_5_1/run.py --check
& D:/anaconda3/envs/simple/python.exe -B D:/MuJoCo/LiuQP_experiment_5_1/run.py
& D:/anaconda3/envs/simple/python.exe -B D:/MuJoCo/LiuQP_experiment_5_1/evaluate_round2.py
& D:/anaconda3/envs/simple/python.exe -B D:/MuJoCo/LiuQP_experiment_5_1/view.py --check
```

使用原32逻辑CPU配置。每次运行保存独立时间戳目录，不读取原基准轨迹。`evaluate.py`、`write_report.py`、`make_preview.py`是首版离线脚本；当前报告使用evaluate_round2.py，不要用首版脚本覆盖当前报告。

## 方案与覆盖范围

首版使用完整5mm格子外包；当前改为每个观测体素已认证的连续三维测量支持体，消除额外格子膨胀。没有丢测量，没有降低6mm余量。核心Q包含测量包络，额外U=0，MVT/AABB后进入原球—椭球精确距离分支。不是TSDF或未知物体完整实心重建。

- [开发协议](docs/开发协议.md)
- [轮2覆盖对象与公式](docs/轮2连续占据支持体协议.md)
- [实际实现](src/v43_ellipsoid/continuous_volume.py)
- [验证数据](validation/evaluation_round2.json)

各prepare/patch/bootstrap脚本用于历史移植，不应重复执行。当前代码已接入，直接运行run.py。

## 5.1 配对对照实验（新增）

两组对照均在 [comparisons](comparisons) 中，使用同一冻结观测地图进行几何消融。第一组同数量椭球／外接球，第二组认证多球覆盖（2、1、0.5 mm外扩）。

- [配对对照结果报告](comparisons/结果报告.md)
- [实验协议](comparisons/plan/experiment-protocol.md)
- [汇总表](comparisons/tables/comparison.csv)
- [闭环结果](comparisons/tables/rollouts.csv)

这是冻结地图对照，和上方在线达标实验的感知条件不同，耗时不能直接混合比较。原在线运行入口和达标记录保持原样。
