# 原 v4.3 椭球与原 v4.4 球版

本文件夹只服务于你认可的两组实验。**以原实现和原记录为基准，不以新重跑结果替换它们。**不包含其他实验场景、路径规划器、恢复工作台或独立重写版。

## 直接查看原结果

双击 `查看原椭球版.cmd` 或 `查看原球版.cmd`。这两个入口默认读取本目录 `baselines/` 中的原记录，不启动新实验。

| 原基准 | 最终误差 | 最后代理数 | 在本对照中的作用 |
|---|---:|---:|---|
| v4.3 椭球 | 0.176859684 mm | 2187 | 末端到达 |
| v4.4 球 | 263.723994966 mm | 3697 | 保留原受阻对照，不是声称球版到达 |

MuJoCo 窗口内：`1` 球版，`2` 椭球版；`空格` 暂停；`N` 单周期；`0` 从头；`F` 末态；`V` 隐藏/显示全部证书叠层（不隐藏机械臂本体）；`P` 点云；`O` 全部障碍代理；`Q` 当前实际进入 QP 的代理；`R` 机械臂证书球；`H` 限制运动的球—障碍对；`C/E/U` 核心椭球/外包络/不确定性叠层。

原记录未持久化粗筛候选 ID，所以 `B` 层为空，不冒充已经记录；`O` 和 `Q` 使用真实原记录。绿色点是当时已发布的 CenterVox 代表点，为显示性能最多绘制约 3000 点，控制和证书数据没有因此降采样。完整代理数不裁减。

## 从头运行一次

双击 `运行椭球版.cmd` 或 `运行球版.cmd`。两者都是新一轮真实在线感知和控制，固定执行原 3000 周期，不读取原轨迹作指令。新数据存到 `runs/v43_ellipsoid/` 或 `runs/v44_sphere/` 的独立时间戳目录，绝不覆盖原基准。

命令行等价入口：

```powershell
& D:\anaconda3\envs\simple\python.exe -B D:\MuJoCo\LiuQP_v43_v44\run.py ellipsoid
& D:\anaconda3\envs\simple\python.exe -B D:\MuJoCo\LiuQP_v43_v44\run.py sphere
```

两个实验应分别运行，不要为了比较实时耗时而同时运行。默认查看入口始终看原基准；需要看新结果时，使用 `view.py --sphere-run <新球结果目录> --ellipsoid-run <新椭球结果目录>`，路径应指向包含 `summary.json` 的目录。

## 文件夹内有什么

```text
LiuQP_v43_v44/
├─ README.md                         本说明
├─ 查看原球版.cmd / 查看原椭球版.cmd   原记录的 MuJoCo 查看入口
├─ 运行球版.cmd / 运行椭球版.cmd       新在线运行入口
├─ run.py / view.py                  两个简洁 Python 入口
├─ configs/original.json             原正式参数
├─ assets/                           唯一抽屉 XML、场景元数据及模型许可证
├─ src/common/                       两版共用的原控制、相机、碰撞、地图及查看模块
├─ src/v43_ellipsoid/                原椭球构造器和原异步闭环
├─ src/v44_sphere/                   原自适应不可约球构造器和原异步闭环
├─ native/                          原 C++ 碰撞库源码与预编译 DLL
├─ baselines/v43_ellipsoid/          原椭球记录的精确副本
├─ baselines/v44_sphere/             原球记录的精确副本
├─ runs/                            本次两组打包回归及今后自行运行的结果
├─ tests/                           仅用于这两版的原记录回归和完整性检查
└─ docs/                            仅这两版的协议、代码导读、核验和来源清单
```

`build_native.cmd` 和 `requirements.txt` 也是这两组的构建/环境依赖。恢复工具、历史补丁、其他版本工程全部在本目录之外；本目录的运行不调用它们。不是从别处加载旧模块的“入口壳”。

## 环境与核验

当前启动脚本使用已验证的 `D:\anaconda3\envs\simple\python.exe`，Python 3.11.9；库版本见 `requirements.txt`。原 CPU 亲和配置针对本实验的 32 逻辑 CPU 机器；DLL 需要 Windows x64、AVX2 和相应 VC/OpenMP 运行库。移动本文件夹后项目内部路径仍有效；更换 Python 安装位置时只需修改四个启动脚本的解释器路径。

已有可运行 DLL，无需先编译。确需重编译时，`build_native.cmd` 使用原 MSVC 2017 参数；不要在实验或查看器占用 DLL 时重编译。原 DLL 文件未找回，提供的是按原 C++ 重编译的二进制，源码和新 DLL 哈希分别保存，不冒称原 DLL 字节一致。

只检查文件完整性：

```powershell
& D:\anaconda3\envs\simple\python.exe -B D:\MuJoCo\LiuQP_v43_v44\tests\verify_package.py
```

先读 [实验协议](docs/实验协议.md)，再读 [代码导读](docs/代码导读.md)。打包回归、实时性和可观测性的已知边界见 [核验记录](docs/核验记录.md)。目录整理完成不意味着原实验所有研究验收条件已经通过。
