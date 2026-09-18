# 基于增量深度点云的 UR5 狭窄抽屉进入：LiuQP、IRIS 椭球分离与 VCC/MVT 的统一数学模型及无作弊实验协议

**文档性质：**研究方法、数学定义与正式实验协议（实现对齐版）  
**版本：**1.0-formal-v3，2026-08-30  
**适用代码：**`ur5_liuqp_iris_scenes` 的 `formal_drawer_two_camera` 正式实验  
**核心命题：**在真实几何可达且控制器只接收同一个最终目标的条件下，由同一局部环境单元构造的球形障碍代理会封闭必经狭窄通道，使球形 LiuQP 无法到达；使用相同中心、相同数量、相同机器人证书和相同安全裕量的椭球障碍代理仍保守覆盖输入，却保留可穿越截面，使椭球 LiuQP 依靠自身逐周期优化到达。IRIS 仅启发椭球支持函数和最近分离平面的推导；VCC-inspired CenterVox、MVT、AABB、SoA/SIMD 只处理覆盖保持与候选查询，不充当路径规划器。

---

## 1. 研究问题、因果假设与结论边界

### 1.1 需要验证的命题

给定固定基座 UR5e、一个已打开的抽屉、初始关节状态 \(q_0\) 和唯一末端目标 \(p_g\)，研究以下因果链：

1. MuJoCo 精确碰撞几何下，存在连续全机械臂无碰路径 \(q:[0,1]\rightarrow\mathcal Q_{\mathrm{free}}^{\mathrm{true}}\)，满足 \(q(0)=q_0\) 且末端进入目标球 \(\|p_{ee}(q(1))-p_g\|\le\varepsilon_g\)。
2. 从逐帧深度观测得到同一组带误差半径的表面代表点，并用同一分簇 \(\{\mathcal C_j\}_{j=1}^{N_p}\) 构造两套环境证书。
3. 球证书在所有通往目标球的必经截面上形成连续占据屏障。因此球形 LiuQP 的失败不是有限迭代、目标太远、初值偶然、求解器异常或轻微扰动可解除的局部停滞，而是代理几何造成的结构性不可达。
4. 椭球证书对相同观测保持保守包络，却因各向异性支持半径更小而在相同截面上保留正净空。没有路点、路径、随机扰动或区域链时，椭球 LiuQP 到达目标。
5. 采用 VCC-inspired 在线点云过滤、MVT/AABB/SIMD 粗查询及 LiuQP ordered erase-remove 后，候选、QP 行和轨迹与相同 AABB 判据下的暴力参考实现一致，并满足预先声明的实时周期。

### 1.2 允许人为规定什么

允许研究者规定机器人型号、抽屉尺寸、打开距离、相机安装位姿、初始构型、最终目标、控制频率、安全裕量和传感器噪声模型。场景尺寸必须在实验前冻结，并同时接受“真实几何可达”和“代理屏障”审计。

### 1.3 严禁的作弊式改进

控制器不得读取或使用：

- 人工路点、人工参考轨迹、分阶段目标、演示轨迹或最终成功轨迹；
- A*、RRT、PRM、GCS、轨迹优化器或 IRIS 区域连接链生成的在线引导；
- MuJoCo 中完整障碍物的真值位姿、真值网格、接触距离或未来深度帧；
- 预先融合的完整点云；
- 隐藏机器人后让深度射线穿过机器人得到的“透视”点云；
- 随机目标抖动、Brownian motion、人工排斥脉冲、失败后重置姿态；
- 球版与椭球版不一致的点云、分簇、代理数量、机器人碰撞证书、安全裕量、速度上限、QP 权重或成功阈值；
- 将长探杆预先伸入抽屉、缩小机器人碰撞体或扩大抽屉以制造成功；
- 仅报告 QP 求解器时间并称为完整控制周期实时性。

旧的 `prescribed` 结果和“500 mm 长管起始已进入抽屉”的结果仅保留为诊断记录，不可作为主实验结果。

---

## 2. 符号、离散时间与控制器信息集

设 UR5e 有 \(n=6\) 个关节，控制周期 \(\Delta t\)，时刻 \(t_k=k\Delta t\)。

| 符号 | 定义 |
|---|---|
| \(q_k,\dot q_k\) | 当前关节位置与本周期命令速度 |
| \(p_{ee}(q),J_{ee}(q)\) | 末端位置与平移雅可比 |
| \(p_i(q),J_i(q),r_i\) | 第 \(i\) 个机器人球证书的中心、雅可比、半径 |
| \(D_k^{(c)},S_k^{(c)}\) | 第 \(c\) 个局部相机在第 \(k\) 帧的深度图与逐像素几何分割图 |
| \({}^WT_{C_c}(q_k)\) | 第 \(c\) 个相机坐标系到世界坐标系变换 |
| \(\mathcal P_k\) | 截止时刻 \(k\) 的增量融合点云 |
| \((\hat p_m,\rho_m)\) | CenterVox 代表点及其保守误差球半径 |
| \(\mathcal C_j\) | 第 \(j\) 个共同点簇 |
| \(B(c,r)\) | 中心 \(c\)、半径 \(r\) 的闭球 |
| \(E(c,Q)\) | \(\{x:(x-c)^TQ^{-1}(x-c)\le1\}\)，\(Q\succ0\) |
| \(\delta_s\) | 统一安全裕量 |
| \(\varepsilon_g\) | 末端目标成功半径，主实验固定为 18 mm，并需连续保持 10 个周期 |

控制器在周期 \(k\) 的允许信息严格定义为

\[
\mathcal I_k=\{q_{0:k},D_{0:k}^{(1:N_c)},S_{0:k}^{(1:N_c)},
{}^BT_{C_{1:N_c}},\mathcal K,p_g,\Theta_{\rm fixed}\},
\tag{1}
\]

其中 \(\mathcal K\) 是公开的 UR5e 运动学与自身几何，\(\Theta_{\rm fixed}\) 是实验前冻结参数。任何依赖 \(D_{k+1:}\) 或环境真值几何的量都不得进入控制器。评估器可读取真值，但必须在独立进程/模块中，且其输出不得回流。

---

## 3. 抽屉场景与真实几何可达性

### 3.1 场景拓扑

UR5e 基座位于抽屉正面下方。初始时末端、腕部和前臂均在抽屉开口下方且在抽屉外。目标位于已打开抽屉内部深处；成功不仅要求末端进入目标球，还要求腕部和一段前臂实际跨过抽屉前沿平面。正式场景不安装夹爪、长杆、探针或虚拟工具，控制目标是 UR5 末端 attachment 中心。

定义抽屉前沿平面 \(\Pi_f=\{x:n_f^Tx=b_f\}\)，目标位于内侧。任意从 \(q_0\) 到目标集合的连续机械臂路径都必须使指定机器人证书中心穿过一个有界开口截面 \(\Sigma\subset\Pi_f\)。\(\Sigma\) 因而是可审计的“必经截面”，而不是凭轨迹观察事后挑选的位置。

### 3.2 真实几何可达证书只用于评估

真实可达性由三个互不回流控制器的证据共同验证：目标点至少存在一个精确 MuJoCo 无碰撞 IK；移除环境代理后，同一 final-goal-only LiuQP 可从相同初态到达；正式椭球 LiuQP 的实际逐周期轨迹到达目标，并由 MuJoCo 接触日志和连续代理 sweep 重放确认。数学上写为

\[
\exists q(s),\quad
q(0)=q_0,\quad
\|p_{ee}(q(1))-p_g\|\le\varepsilon_g,
\quad d_{\rm true}(q(s))\ge d_{\rm audit}>0,\ \forall s\in[0,1].
\tag{2}
\]

正式门槛禁止 RRT、A*、PRM、人工路径或其他规划器生成存在性见证。IK 只证明目标构型本身可达；整条连续运动的证据必须来自椭球 LiuQP 自己产生的轨迹。后处理审计只能读取已完成运行，不能接受、缩放、回退或改写任一在线命令。

### 3.3 公平机器人证书

主对照中机器人始终使用同一套 LiuQP 球证书

\[
\mathcal R(q)=\bigcup_{i=1}^{N_r}B(p_i(q),r_i).
\tag{3}
\]

球版与椭球版的差别只在环境代理。机械臂椭球证书可作为独立附加消融，不得混入主因果比较。

---

## 4. 局部机载深度视角与逐帧增量地图

### 4.1 物理安装、局部可观测性与成像参数

环境不是预先全局已知，也不假定某个障碍物永久不可见。机器人只能像人在房间中一样，从当前机载视角看见无遮挡的局部表面；随着手臂移动、转腕或不同机载视角获得新视线，地图才逐帧扩展。桌后、墙缝、抽屉遮挡面等尚未被任何射线观测的区域必须继续标为 `unknown`，不得由 MuJoCo 真值补齐。

相机集合可由一个腕部视角，或现实可安装的腕部、末端、前臂补充视角组成。数量和外参不先验固定为“一台”或“九台”，而由独立可观测性审计确定：若目标相关工作区存在因自遮挡或视场造成的永久盲区，可以调整相机外参或增加少量补充视角；每个相机都必须具有外壳碰撞几何、有效视场和量程，光心不得位于机器人或抽屉实体内部。建议按 Intel RealSense D405 的真实量级建模：单机外形约 \(42\times42\times23\) mm、最小有效距离 70 mm、典型深度视场 \(87^\circ\times58^\circ\)。所有外壳必须确实能安装在所选连杆上且不得相互穿透。

相机布局的选择只允许使用视场覆盖率、永久盲区和物理安装可行性，不得根据球版或椭球版的成功轨迹调参。布局一经选定，必须在正式对照前写入冻结配置；球版和椭球版使用完全相同的相机模型、频率、延迟、噪声种子和首帧。闭环轨迹分叉后，两版相机位姿及后续深度帧自然不同，必须分别由各自当前构型因果采集，不能把一版未来看到的帧复制给另一版。若日志出现“9 views”，它可以表示同一相机在九个时刻的观测，也可以表示已冻结阵列在若干时刻的观测，但绝不表示九个全局真值视点。

每台相机固定于机器人连杆，MuJoCo 每步通过正运动学更新其世界位姿。相机视角的变化必须来自 LiuQP 正常命令导致的机械臂运动；不得为建图插入隐藏扫描路径、额外路点或绕开控制器的转腕命令。MuJoCo 相机前向为局部 \(-z\) 轴。分辨率、帧率、近远裁剪面、噪声、延迟和外参误差在配置文件冻结；主实验建议控制 50 Hz、深度 30 Hz 异步更新，控制线程只使用最近一份已经完整提交的因果地图快照。

本研究不把“主动寻找最佳视角”作为待比较算法，也不要求机器人在任一时刻拥有全局地图。实验前可以为消除永久盲区而调整或增加物理可安装视角；正式冻结后，地图只能随正常闭环运动逐帧增长。即使长时间运行后累计地图覆盖了房间的大部分区域，控制器在第 (k) 周期仍只能使用截至该周期已经观测并发布的局部累积信息。本文真正比较的是：同一类局部深度信息被更新后，球形 LiuQP 与椭球型 LiuQP 如何把同源点云变成不同几何证书、构造约束并指导末端到达唯一目标。

### 4.2 深度反投影

对第 \(c\) 个相机的像素 \((u,v)\)、深度 \(z_{uv}^{(c)}\)，相机内参为 \((f_x^{(c)},f_y^{(c)},c_x^{(c)},c_y^{(c)})\)，有

\[
{}^{C_c}p_{uv}=z_{uv}^{(c)}
\begin{bmatrix}
(u-c_x^{(c)})/f_x^{(c)}\\(v-c_y^{(c)})/f_y^{(c)}\\1
\end{bmatrix},\qquad
{}^Wp_{uv}^{(c)}={}^WT_{C_c}(q_k){}^{C_c}p_{uv}.
\tag{4}
\]

若渲染器返回的是光轴深度，应使用式 (4)；若返回沿射线距离，则先乘单位射线。两者不得混用，并以平面标定测试验证误差小于 0.5 mm。

### 4.3 遮挡与自身点删除

渲染深度时必须保留机器人和相机外壳，使它们真实遮挡后方环境；同时渲染逐像素 geom ID。得到深度后，利用分割图删除命中机器人 geom 的像素：

\[
\mathcal Z_k=\bigcup_{c=1}^{N_c}\{{}^Wp_{uv}^{(c)}:
D_k^{(c)}(u,v)\text{有效},\ S_k^{(c)}(u,v)\notin\mathcal G_{robot}\}.
\tag{5}
\]

先隐藏机器人再渲染会产生穿透视线，是拒收条件。机器人阴影后方仍为未知，不得因删除自身像素而标成空闲。

### 4.4 三状态占据与信息因果

每条有效射线在测量下界之前形成观测空闲段，在端点误差壳形成占据证据，端点之后保持未知：

\[
\mathcal F_k^{ray}=\{o_k+\tau \hat r:\tau\in[z_{min},z_{uv}-\rho_{uv})\},
\quad
\mathcal O_k^{ray}=B({}^Wp_{uv},\rho_{uv}),
\tag{6}
\]

\[
\mathcal M_k=\operatorname{Fuse}(\mathcal M_{k-1},\mathcal F_k^{ray},\mathcal O_k^{ray}),
\quad \mathcal M_{-1}=\varnothing.
\tag{7}
\]

其中状态为 free / occupied / unknown。“unknown”必须原样保留，既不得补成 free，也不得凭空生成障碍代理。主实验采用“observed_only”策略：LiuQP 和连续步长守卫只使用截至当前已经观测并完成证书构造的 occupied 代理；未知状态保留在地图和日志中，但不把所有未知体素一律当成硬障碍。原因是机载相机必然存在自身外壳近场和自遮挡，一个“全机器人扫掠体必须完全落在 free 体素”策略会被无环境障碍的单个量化边界体素永久锁死，这检验的是传感器布置而不是 LiuQP 表示能力。

可观测性审计中的风险表面不是“欧氏距离落入阈值的所有盒体表面采样点”。对于薄板，实体本身遮挡的反面不是当前最近接触见证面，也不应要求相机同时看见。对每个机器人证书与每个实体，只审计达到点到该实体真实距离的最近表面见证片，并用表面采样间距的 Lipschitz 修正保守扩张。审计使用源帧时刻加实测地图/代理发布延迟；不能用采集时刻冒充控制器真正得到该信息的时刻。

为避免“observed_only”被误解为把未知当作安全，必须增加只读真值可观测性门。令 \(t_{\rm risk}(x)\) 为真实障碍表面点 \(x\) 首次进入机器人证书的预定观测/制动距离 \(d_{\rm obs}\) 的时刻，\(t_{\rm seen}(x)\) 为任一局部相机首次把 \(x\) 的保守表面片写入已发布代理的时刻，要求

\[
t_{\rm seen}(x)+\tau_{\rm pub}\le t_{\rm risk}(x)-\tau_{\rm guard},
\qquad
x\in\mathcal O_{\rm proxy}\bigl(t_{\rm risk}(x)-\tau_{\rm guard}\bigr).
\tag{7a}
\]

该评估器可以读取 MuJoCo 真值来判定一次运行是否应以“unobserved_hazard”拒收，但其结果不得回流控制、相机转动、速度缩放或代理生成。若失败，应在正式冻结前调整/增加现实可安装的腕部、末端或前臂视角并重跑两版；不得由真值直接补点。“strict_unknown”（所有扫掠体素必须为 free）保留为安全极限消融，不是球/椭球主因果对照。

点云的因果更新为

\[
\mathcal P_k=\mathcal V\!F\left(\mathcal P_{k-1}\cup\mathcal Z_k\right),
\tag{8}
\]

不允许在 \(k=0\) 直接调用场景表面采样函数生成完整书架点云。

### 4.5 单点误差球

像素脚印、深度噪声、外参与时间配准误差合成保守半径

\[
\rho_{uv}=\rho_z(z)+z\tan\frac{\Delta\theta_{pix}}{2}
+\rho_{ext}+v_{max}\tau_{sync}+\rho_{cal},
\tag{9}
\]

其中各项取预先标定的上界而非标准差；若只给高斯标准差，则须声明置信倍数并报告失效概率。

---

## 5. VCC-inspired CenterVox 与保守表面覆盖

### 5.1 中心选择

过滤体素边长为 \(L_f\)，原点 \(o_f\)。点 \(p\) 的整数键为

\[
v(p)=\left\lfloor\frac{p-o_f}{L_f}\right\rfloor.
\tag{10}
\]

对同一键内点集 \(\mathcal V_m\)，保留最接近体素中心 \(c_m^v=o_f+(v_m+\tfrac12\mathbf1)L_f\) 的点

\[
\hat p_m=\arg\min_{p\in\mathcal V_m}\|p-c_m^v\|_2.
\tag{11}
\]

这与 VCC 的 CenterVox 选择思想一致，但安全性不能只依赖“点在中心附近”。被删除点的误差球必须由代表点继承：

\[
\hat\rho_m=\max_{p_a\in\mathcal V_m}
\left(\|p_a-\hat p_m\|_2+\rho_a\right).
\tag{12}
\]

于是 \(\bigcup_{a\in\mathcal V_m}B(p_a,\rho_a)\subseteq B(\hat p_m,\hat\rho_m)\)。该式替代简单的 \(\sqrt3L_f/2\) 固定膨胀，既精确记录实际残差，又不会漏掉角落点。

### 5.2 连续表面而非离散点的覆盖

点云只采样表面。对每个观测三角像素片或局部平面片 \(S_m\)，验证采样网格最大间距 \(h_m\)，则距离函数的 1-Lipschitz 性给出

\[
\sup_{x\in S_m}\operatorname{dist}(x,\hat{\mathcal P})
\le d^{sample}_{m,max}+\frac{\sqrt2}{2}h_m.
\tag{13}
\]

式 (13) 的右端加入 \(\hat\rho_m\) 或代理 offset。论文必须报告连续覆盖上界，不能以“图上球看起来相交”代替包络证明。

---

## 6. 相同分簇下的球与椭球环境代理

### 6.1 公平分簇

过滤后的带半径点 \(\{(\hat p_m,\hat\rho_m)\}\) 仅按空间连通性与局部法向相似性分簇。法向由邻域协方差最小特征向量估计；不输入“这是竖直墙”或世界 \(z\) 轴。分簇结果 \(\mathcal C_j\)、中心 \(c_j\) 和代理数量 \(N_p\) 在球版与椭球版完全相同。

### 6.2 球代理

对簇 \(\mathcal C_j\)，中心固定为共同中心 \(c_j\)，球半径为

\[
R_j=\max_{m\in\mathcal C_j}
\left(\|\hat p_m-c_j\|_2+\hat\rho_m\right)+\rho^{cover}_j.
\tag{14}
\]

环境球集为 \(\mathcal O^S=\bigcup_j B(c_j,R_j)\)。大而近似平面的簇会使切向跨度也进入各向同性半径，这是预期要检验的保守性来源，不得人为减小半径。

### 6.3 椭球核心与 Minkowski 误差球

对相同簇计算 PCA 正交基 \(U_j\)，局部坐标 \(y_m=U_j^T(\hat p_m-c_j)\)。先取正则化种子半轴 \(a^0_{j\ell}\ge a_{min}\)，再用最大 Mahalanobis 半径统一缩放：

\[
\gamma_j=\max_{m\in\mathcal C_j}
\sqrt{\sum_{\ell=1}^3\frac{y_{m\ell}^2}{(a^0_{j\ell})^2}},\qquad
a_{j\ell}=\gamma_ja^0_{j\ell},
\tag{15}
\]

\[
Q_j=U_j\operatorname{diag}(a_{j1}^2,a_{j2}^2,a_{j3}^2)U_j^T.
\tag{16}
\]

测量与表面覆盖误差不通过把每条轴简单加长来处理，而保留为各向同性 Minkowski offset

\[
\eta_j=\max_{m\in\mathcal C_j}\hat\rho_m+\rho^{cover}_j,
\quad
\mathcal O_j^E=E(c_j,Q_j)\oplus B(0,\eta_j).
\tag{17}
\]

逐点验证 \((\hat p_m-c_j)^TQ_j^{-1}(\hat p_m-c_j)\le1+\epsilon_{num}\)，并用式 (13) 验证连续表面覆盖。这样既完全包络已观测障碍表面及误差，又保留薄板/墙面法向上的小支持半径；不存在 `ellipsoid-fast` 近似分支。

---

## 7. VCC-inspired MVT、AABB 与 SIMD 粗阶段

### 7.1 数据流

在线数据流固定为

\[
D_k,S_k\rightarrow\text{增量融合}\rightarrow\text{CenterVox}
\rightarrow\text{共同分簇/代理}\rightarrow\text{MVT}
\rightarrow\text{global AABB}\rightarrow\text{local AABB}
\rightarrow\text{SIMD 粗筛}\rightarrow\text{精确窄阶段}\rightarrow\text{QP}.
\tag{18}
\]

MVT 只索引截止当前时刻的代表点或代理 AABB。动态更新使用双缓冲快照：感知线程构建下一版本，完成后以原子指针切换，控制周期不会读取半构建结构。

### 7.2 自适应体素尺寸

VCC 对最大查询球使用 \(L=r_{max}+r_{point}\)，使查询最多访问 \(3^3=27\) 个体素。本项目机器人查询仍为球，故定义

\[
L_q=r_{max}+\eta_{max}+v_{center,max}\Delta t,
\quad
G=\left\lfloor\frac{W_{width}}{L_q}\right\rfloor.
\tag{19}
\]

若不同机器人球半径跨度很大，可按半径层级建立多个 MVT，但每层查询范围必须包含扫掠半径与误差 offset。不得为了保持 27 个体素而漏查跨层代理。

### 7.3 三层稀疏表与 SoA

体素键 \((v_x,v_y,v_z)\) 通过连续 Table Pool 的 X 表、Y 表、Z 表三个 offset 定位，非空体素的点/代理坐标以 SIMD 对齐的 Structure-of-Arrays 保存：

\[
x=[x_1,\ldots,x_P],\quad y=[y_1,\ldots,y_P],\quad z=[z_1,\ldots,z_P].
\tag{20}
\]

不足一个 SIMD lane 的末尾以 \(+\infty\) 填充。Python 的嵌套 `dict` 原型只能称为“逻辑 MVT”，不能作为 VCC-style MVT/SIMD 性能结论；最终实时实验必须使用 C++ 连续内存池和 AVX2（x86）或 NEON（ARM）内核。

### 7.4 保守候选条件

机器人球在一个周期内的中心扫掠 AABB 半边长上界为

\[
h_i=r_i+\delta_s+|J_i|\bar{\dot q}\Delta t+\rho_{map},
\tag{21}
\]

其中绝对值逐元素计算。球代理 AABB 为 \(R_j\mathbf1\)；椭球 offset 代理 AABB 半边长为

\[
h_j^E=\sqrt{\operatorname{diag}(Q_j)}+\eta_j\mathbf1.
\tag{22}
\]

只有当机器人扫掠 AABB 与代理 AABB 相交时进入窄阶段。global AABB、体素 AABB 和 local point AABB 均只能造成“多取”，不能造成“漏取”。以无索引全枚举为 oracle，所有可能在一步内接触的 pair 必须满足

\[
\mathcal C^{oracle}_k\subseteq\mathcal C^{MVT}_k.
\tag{23}
\]

SIMD 只批量计算平方距离、AABB 相交和支持函数初筛，不用球—点公式替代椭球精确最近点。

---

## 8. 球与椭球的精确最近点、支持函数和分离平面

### 8.1 球代理

机器人中心 \(p_i\) 到环境球 \(B(c_j,R_j)\) 的方向、近侧切点和安全净空为

\[
n_{ij}=\frac{c_j-p_i}{\|c_j-p_i\|},\quad
z_{ij}=c_j-R_jn_{ij},\quad
g^S_{ij}=\|c_j-p_i\|-R_j-r_i-\delta_s.
\tag{24}
\]

安全半空间为 \(n_{ij}^Tx\le n_{ij}^Tz_{ij}-r_i-\delta_s\)。

### 8.2 椭球支持函数

椭球 \(E(c,Q)\) 沿单位方向 \(n\) 的支持函数为

\[
h_E(n)=n^Tc+\sqrt{n^TQn},
\quad
\min_{x\in E}n^Tx=n^Tc-\sqrt{n^TQn}.
\tag{25}
\]

对 offset 椭球 \(E\oplus B(0,\eta)\)，支持函数再加 \(\eta\)。这正是“沿法向的椭球半径”，但法向不能预先取中心连线后就停止；精确最近点法向需由下一节求得。

### 8.3 点到椭球的 KKT 推导

令 \(Q=U\operatorname{diag}(a_1^2,a_2^2,a_3^2)U^T\)，\(y=U^T(p-c)\)。对椭球外点求

\[
\min_x\frac12\|x-y\|^2\quad
\text{s.t.}\quad\sum_{\ell=1}^3\frac{x_\ell^2}{a_\ell^2}=1.
\tag{26}
\]

拉格朗日函数的一阶条件给出

\[
x_\ell-y_\ell+\lambda\frac{x_\ell}{a_\ell^2}=0
\Rightarrow
x_\ell(\lambda)=\frac{a_\ell^2y_\ell}{a_\ell^2+\lambda}.
\tag{27}
\]

代回约束得到唯一单调根

\[
F(\lambda)=\sum_{\ell=1}^3
\frac{a_\ell^2y_\ell^2}{(a_\ell^2+\lambda)^2}-1=0,
\quad \lambda\ge0,
\tag{28}
\]

\[
F'(\lambda)=-2\sum_{\ell=1}^3
\frac{a_\ell^2y_\ell^2}{(a_\ell^2+\lambda)^3}<0.
\tag{29}
\]

因此不需要固定 80 次二分。使用上一周期同一 pair 的 \(\lambda_{k-1}\) 热启动，执行最多 \(N_N=10\) 次受保护 Newton：

\[
\lambda^+=\lambda-F(\lambda)/F'(\lambda),
\tag{30}
\]

若新值离开已维护的根区间或导数病态，则回退到区间中点；最后最多 \(N_B=32\) 次二分达到残差阈值。记录每步 Newton、二分次数及热启动命中率。

世界最近点、由机器人指向障碍物的法向和净空为

\[
z=c+Ux(\lambda^*),\quad
n=\frac{z-p}{\|z-p\|},\quad
g^E=\|z-p\|-r_i-\eta_j-\delta_s.
\tag{31}
\]

等价地，近侧支持平面偏置

\[
\beta=n^Tc-\sqrt{n^TQn}-\eta_j,
\quad n^Tp\le\beta-r_i-\delta_s.
\tag{32}
\]

式 (31) 与式 (32) 的数值差必须在容差内。若机器人中心位于椭球核心内，视为碰撞并采用确定性逃离法向；这种状态不得作为成功起点。

### 8.4 在线椭球与方向不确定性的精确支持和

在线代理不仅含拟合椭球 \(Q_j\)，还含由 CenterVox 覆盖误差、中心迁移和多帧融合得到的方向不确定性椭球 \(U_j\succeq0\) 以及标量 offset \(\eta_j\)。对机器人球 \(B(p_i,r_i)\) 和障碍集合

\[
\mathcal O_j=E(c_j,Q_j)\oplus E(0,U_j)\oplus B(0,\eta_j),
\]

单位法向 \(n\) 下的分离余量为

\[
h_{ij}(n)=n^T(c_j-p_i)-r_i-\sqrt{n^TQ_jn}
-\sqrt{n^TU_jn}-\eta_j-\delta_s,
\quad \|n\|=1.
\tag{32a}
\]

真实支持法向由 \(\max_{\|n\|=1}h_{ij}(n)\) 求得，而不是直接把中心连线当作法向。对应的障碍近侧支持点为

\[
z_j(n)=c_j-\frac{Q_jn}{\sqrt{n^TQ_jn}}
-\frac{U_jn}{\sqrt{n^TU_jn}}-\eta_jn,
\quad
h_{ij}(n)=n^T(z_j(n)-p_i)-r_i-\delta_s.
\tag{32b}
\]

实现沿单位球面切空间使用受保护 Riemannian Newton、Armijo 下降和同一机器人球—稳定代理 ID 的上一周期法向热启动。热启动法向和无缓存时的中心连线都必须先归一化到 \(S^2\)。常规预算为 16 次；若 KKT 切向残差超过 \(10^{-7}\)，则从归一化中心连线重启并把保护预算提高到 64 次，重启后仍超限就中止该运行，禁止把不准确法向写入 QP。该问题不是式 (28) 的一维单调根，因此不使用固定二分；式 (28)–(30) 的 Newton/必要二分仅用于无 \(U_j\) 的点到核心椭球真实最近点。两条路径都必须满足残差阈值并与标量参考核对。

### 8.5 与 IRIS 的关系

IRIS 对当前椭球度量 \(E\) 求最近障碍点 \(x_i^*\)，再用 \(a_i=E(x_i^*-c)\) 构造分离超平面，并与最大体积内接椭球交替。本项目借鉴的是“椭球度量最近点—支撑平面”思想，并把它用于工作空间环境代理与速度级 QP；主在线控制不执行 IRIS 的构型空间 MVIE 膨胀，也不建立 GCS 区域链。若未来增加低频 IRIS 区域层，必须作为新方法单独消融。

---

## 9. LiuQP ordered erase-remove 的椭球支持推广

LiuQP 原文所称 erase-remove idiom 是 C++ 容器常用的删除写法，不是一种专门数据结构，也不是 IRIS 模块。候选代理先按与当前机器人球的保守距离下界稳定排序，保留第一个代理及其近侧分离平面，再删除完全位于该平面障碍侧的后续代理；对剩余序列重复此过程。对在线椭球代理，几何后方判据为

\[
\min_{x\in E(c_k,Q_k)\oplus E(0,U_k)\oplus B(0,\eta_k)}n_j^Tx
=n_j^Tc_k-\sqrt{n_j^TQ_kn_j}-\sqrt{n_j^TU_kn_j}-\eta_k
\ge\beta_j-\epsilon_{dom}.
\tag{33}
\]

式 (33) 正是球版中心投影减半径规则的支撑函数推广；球版取 \(Q_k=R_k^2I,U_k=0,\eta_k=0\)。正式实现按该确定性顺序调用 C++ 批量支持核，并记录每个周期删除数。若某 pair 的净空已进入 CONTACT/RECOVERY 区间，即使它被另一分离平面遮蔽，也重新加入自己的硬恢复行，避免新发布的保守代理重叠被普通冗余规则掩盖。

这里不求 Farkas 证书，也不声称 erase-remove 保持“所有候选逐行加入 QP”时的原始可行域；它是两种 LiuQP 表示共同使用的控制器组成。纯 MVT/AABB/SIMD 加速等价性必须固定同一候选 AABB 判据和同一 ordered erase-remove，然后验证候选 ID/顺序、QP 行哈希、\(\dot q\) 与轨迹一致。没有被 MVT 查询到的远处代理属于宽阶段排除，不能记作 erase-remove，更不能称为 IRIS 删除。

---

## 10. LiuQP 的完整速度级二次规划

### 10.1 只给最终目标的任务反馈

没有参考路径时，LiuQP 论文式 (17) 退化为最终目标反馈：

\[
v_k^*=\operatorname{sat}_{v_{max}}
\left(K_p(p_g-p_{ee}(q_k))\right).
\tag{35}
\]

这里没有中间 \(\tilde p(t)\)。任何从预生成轨迹读取 \(\tilde p_k\) 的运行都标为 `prescribed`，不得列入主比较。

### 10.2 目标函数

主 QP 为

\[
\min_{\dot q,s_t}\quad
\frac12\|\dot q\|_{W_q}^2
+\frac12\|J_{ee}\dot q-v_k^*-s_t\|_{W_t}^2
+\frac12\|s_t\|_{W_s}^2
+\frac12\|\dot q-\dot q_{k-1}\|_{W_\Delta}^2
+\sum_{(i,j)\in\mathcal N_k}
\frac{\mu_{ij}}2(n_{ij}^TJ_i\dot q)^2.
\tag{36}
\]

任务跟踪是软目标；\(s_t\) 明确表示任务松弛。姿态保持项若使用，必须在两版保持完全相同且只依赖初始姿态目标。不得给椭球版增加阶段性姿态引导。

### 10.3 关节与工作空间约束

\[
\max\left(\dot q_{min},\frac{q_{min}+\epsilon_q-q_k}{\Delta t}\right)
\le\dot q\le
\min\left(\dot q_{max},\frac{q_{max}-\epsilon_q-q_k}{\Delta t}\right).
\tag{37}
\]

对工作空间半空间 \(a_f^Tx\le b_f\)：

\[
a_f^TJ_i\dot q\le
\frac{b_f-r_i-a_f^Tp_i}{\Delta t}.
\tag{38}
\]

### 10.4 碰撞硬约束

对球或椭球分离平面先定义未扣除安全裕量的证书表面间隙

\[
c_{ij}=\beta_{ij}-r_i-n_{ij}^Tp_i(q_k),
\]

以及硬安全净空 \(g_{ij}=c_{ij}-\delta_s\)。一阶离散预测给出

\[
n_{ij}^TJ_i(q_k)\dot q
\le\frac{\beta_{ij}-r_i-\delta_s-n_{ij}^Tp_i(q_k)}{\Delta t}
=\frac{c_{ij}-\delta_s}{\Delta t}
=\frac{g_{ij}}{\Delta t}.
\tag{39}
\]

式 (39) 是硬约束，不设碰撞 slack。三状态分类必须使用 \(c_{ij}\)，而不是 \(g_{ij}\)：\(c_{ij}\ge d_{near}\) 为 NORMAL，\(0<c_{ij}<d_{near}\) 为 NEAR，只有证书已经接触或重叠的 \(c_{ij}\le0\) 才进入 CONTACT/RECOVERY，并加入 LiuQP 式 (27) 型排斥硬约束

\[
n_{ij}^TJ_i\dot q\le-\gamma_{ij},\qquad\gamma_{ij}>0,
\tag{40}
\]

并删除该 pair 的软近障惩罚，避免目标冲突。正式原文复现冻结接触阈值为零；\(\delta_s=6\,\mathrm{mm}\) 只属于式 (39)，不能把仍有正表面间隙的 pair 提前判成接触。日志必须同时保存 \(c_{ij}\) 与 \(g_{ij}\)。QP 不可行时命令为零并记录“qp_infeasible”；在“strict_unknown”消融中，未知扫掠体素还会触发“safe_stop_unknown”。主实验的“observed_only”不伪造未知约束，而由式 (7a) 的独立门决定该次运行能否作为证据。

### 10.5 OSQP 标准形式与热启动

将式 (36) 展开为 \(\tfrac12x^TPx+q^Tx\)，式 (37)–(40) 组装为 \(l\le Ax\le u\)，使用上一周期 primal/dual 解热启动。求解容差、最大迭代数和失败处理在两版一致。球版解析距离更便宜；椭球版额外耗时来自每个窄阶段 pair 的式 (28)–(30)、支持函数和矩阵运算，而不是 QP 维数必然增加。

### 10.6 不回流控制的连续 sweep 事后审计

正式在线循环直接执行 QP 给出的 \(q^+=q_k+\dot q\Delta t\)，不根据代理 sweep 或 MuJoCo 真值缩放、回溯、接受或拒绝命令。运行结束后，审计器读取已保存的 \(q_k,q_{k+1}\) 和该周期实际因果代理快照，对每个机器人球中心线段验证

\[
\min_{\tau\in[0,1]}g_{ij}(q_k+\tau\alpha\dot q\Delta t)\ge0
\tag{41}
\]

对起点已处于保守代理重叠的 pair，CONTACT/RECOVERY 语义要求运动不加深重叠并沿分离方向恢复；这类 pair 单独记录，不能把负起始净空伪报成新碰撞。审计结果只写入 `post_control_sweep_audit.csv`，不得修改轨迹。这样连续验证仍能检查速度级线性化是否产生漏碰，又不会成为帮助椭球组成功的第二控制器。

---

## 11. “球一定过不去”的连续屏障证明

### 11.1 不是观察停滞，而是拓扑分离

令 \(\Sigma\) 是指定机器人球中心从抽屉外到目标必须穿过的紧致二维截面。球代理对该机器人球的带符号净空函数为

\[
\phi_S(x)=\min_j\left(\|x-c_j\|-R_j-r_i-\delta_s\right).
\tag{42}
\]

每个距离函数是 1-Lipschitz，有限个函数的最小值仍是 1-Lipschitz。在截面三角网格顶点 \(x_m\) 上计算 \(\phi_S(x_m)\)。若最大网格单元直径为 \(h_\Sigma\)，并满足

\[
\max_m\phi_S(x_m)+h_\Sigma< -\epsilon_{barrier},
\tag{43}
\]

则对所有 \(x\in\Sigma\) 有 \(\phi_S(x)<0\)，整个截面都被球证书封闭。若使用单元中心采样，可把 \(h_\Sigma\) 换成单元覆盖半径。目标 18 mm 球的所有入口前方必须至少存在一个满足式 (43) 的截面。

### 11.2 排除“小扰动就能出来”

式 (43) 给出负裕量 \(\epsilon_{barrier}\)。只要场景、点云和代理参数扰动引起的 Hausdorff 变化小于 \(\epsilon_{barrier}/2\)，闭塞仍成立。因此小目标扰动、初始关节扰动或 QP 线性项扰动不能穿过屏障。额外运行多初值/多微扰试验只是统计佐证，不能代替式 (43)。

### 11.3 椭球通道开放证书

椭球 offset 代理的净空为

\[
\phi_E(x)=\min_j\left(\operatorname{dist}(x,E(c_j,Q_j))-\eta_j-r_i-\delta_s\right).
\tag{44}
\]

需要在同一截面找到连通曲线 \(\Gamma\subset\Sigma\) 使

\[
\min_{x\in\Gamma}\phi_E(x)\ge\epsilon_{open}>0,
\tag{45}
\]

并用 1-Lipschitz 网格修正验证连续正净空。式 (45) 只证明代理没有堵路；真正“椭球 LiuQP 可达”还必须由无路点在线运行成功证明。

---

## 12. 实时计算定义与正确性等价

### 12.1 计时边界

主控制目标 50 Hz，周期预算 20 ms。记录同一单调时钟下的：

1. 深度获取/反投影；
2. 增量融合与 CenterVox；
3. 共同分簇/代理局部更新；
4. MVT 更新或快照切换；
5. global/local AABB 与 SIMD 粗筛；
6. 球解析或椭球精确窄阶段；
7. LiuQP ordered erase-remove；
8. QP 组装与 OSQP 求解；
9. MuJoCo 步进与命令提交；
10. 控制线程完整计算周期。

感知按相机请求频率异步运行，代理拟合和索引构建不占用控制线程；分别报告控制计算周期、感知流水线、实际发布频率和快照年龄。冻结的实时门槛为控制计算 p99 不超过 20 ms。另报 mean、median、p95、p99、max 和所有 deadline miss；通过 p99 门槛不等于硬实时，若存在单周期超限必须明确披露。

### 12.2 参考实现与加速实现

纯查询加速使用三个严格等价实现：`AABB-Brute` 对全部代理执行同一 float32 inclusive AABB 判据；`MVT-Scalar` 使用 5 层、每层 27 邻域的连续内存索引和标量 AABB 核；`MVT-AVX2` 保持同一索引与判据，仅把 SoA AABB 比较改为 8-lane AVX2。三者后续使用相同精确窄阶段、ordered erase-remove 和 QP。

对同一输入快照，候选 ID 及顺序、最终 QP 行集合哈希、pair 状态、\(q\)、\(\dot q\)、末端位置和误差必须完全一致。原始 `full_scan` 不施加 action-distance AABB 判据，会把远处代理送入 erase-remove，因此求解的是不同 QP 行集合，只作为“未引入 VCC 宽阶段”的算法基线，不能用于计算纯数据结构加速比。

理论上，MVT 查询在固定 \(K\) 个相邻体素、每体素最多 \(P\) 个点、SIMD 宽度 \(w\) 时为 \(O(K\lceil P/w\rceil)\)；但“常数时间”只在 \(K,P\) 有明确上界时成立。在线新增点导致的更新成本也必须计入。

---

## 13. 实验矩阵、通过条件与日志契约

### 13.1 主矩阵

| 编号 | 感知 | 环境代理 | 查询实现 | 作用 |
|---|---|---|---|---|
| A0 | 真值几何，仅评估器 | MuJoCo 精确几何 | 独立审计 | 证明物理可达，不控制 |
| K1 | 已知冻结局部单元 | 200 个匹配球 | 原始 full scan + LiuQP erase-remove | 静态表示失败基线 |
| K2 | 同 K1 | 200 个匹配椭球 | 同 K1 | 静态表示成功对照 |
| O1 | 两个移动局部深度视角 | CenterVox 匹配球 | 5 层 MVT+AABB+AVX2 | 正式在线球组 |
| O2 | 同 O1，各自因果采帧 | CenterVox 匹配椭球+方向不确定性 | 同 O1 | 正式在线椭球组 |
| X1 | 冻结 K1/K2 输入 | 同一代理 | AABB-Brute / MVT-Scalar / MVT-AVX2 | 查询严格等价性与加速 |

另做球代理数量/分辨率消融（例如由细到粗四档，实际数量由同一算法产生，不手工指定中心）和椭球 204 代理目标档；若最终数量不是 204，应报告实际值，不能为凑数破坏共同分簇。最关键主对照必须使用相同 \(N_p\)。

### 13.2 成功、停滞与失败

成功需同时满足：

\[
\|p_{ee}-p_g\|\le18\text{ mm},\quad
d_{true,min}>0,\quad
d_{proxy,min}\ge0,
\tag{46}
\]

并保持连续 \(T_{hold}\)；指定腕部和前臂确实跨过前沿平面。停滞定义为滑动窗口内目标误差下降小于阈值且 \(\|\dot q\|\) 小，但只作行为描述。球版的强失败结论必须由式 (43) 支撑。QP 不可解、未知区安全停止、碰撞、超时和数值异常分别编码，不得合并成“失败”。

### 13.3 参数冻结与重复性

场景搜索可在开发集进行，但一旦选择主场景，须写入 `frozen_protocol.yaml` 并计算 SHA-256。之后不得基于测试结果调整几何、目标、裕量或权重。至少用多个深度噪声种子和小范围初始关节扰动运行；所有失败均保留。场景设计脚本、XML、相机外参、点云快照、代理文件和可执行版本哈希随结果保存。

每周期日志至少包含：时间戳、地图版本/年龄、相机位姿、原始/过滤点数、代理数、MVT 单元和内存量、候选 pair、精确 pair、删除数、Newton/二分次数、QP 行数/迭代/状态、阶段耗时、\(q,\dot q,p_{ee}\)、目标误差、代理/真值净空和事件码。事后报告另存候选重建、三态地图重放和连续 sweep 结果，并明确其未回流控制。

### 13.4 最小验收清单

- [ ] 初始末端、腕部和前臂均未进入抽屉；无夹爪、长探杆或虚拟工具。
- [ ] 相机布局通过永久盲区/物理安装审计后冻结；每个视角均为具有物理尺寸的腕部、末端或前臂 RGB-D 相机。两版使用相同传感器参数、首帧和噪声种子，轨迹分叉后各自因果采帧。
- [ ] 首帧看不到被遮挡的完整抽屉；点云随运动增长。
- [ ] 自身几何参与遮挡，之后才按 segmentation 删除自身点。
- [ ] A0 以目标无碰撞 IK、无障碍 LiuQP 可达和正式椭球 LiuQP 实际轨迹共同证明物理可达，不生成审计路径。
- [ ] 球与椭球在每个各自快照内使用共同分簇、共同中心、相同代理数量、同机器人球和同 QP 参数；闭环分叉后的不同观测不得跨版本复制。
- [ ] 式 (7a) 的只读评估器对两版均无“unobserved_hazard”；真值结果未回流控制器。
- [ ] 式 (43) 证明球代理连续封闭，式 (45) 证明椭球代理保留正净空。
- [ ] 球型在全部规定微扰中不可达；椭球型无路点到达。
- [ ] MVT/SIMD 无漏候选，AABB-Brute、MVT-Scalar 与 MVT-AVX2 的 QP 行与轨迹等价。
- [ ] 正式在线组控制计算 p99 满足 20 ms，并单独披露最大值和超限周期。
- [ ] 所有旧实验原样保留，新结果写入独立目录。

---

## 14. 模块接口与伪代码

### 14.1 模块接口

| 模块 | 输入 | 输出 | 不允许访问 |
|---|---|---|---|
| `DepthSensor` | \(q_k\)、MuJoCo 渲染缓冲 | \(D_k,S_k,T_{WC}\) | 未来帧 |
| `IncrementalMap` | 上一地图、当前帧 | free/occupied/unknown、点误差球 | 场景真值几何 |
| `CenterVox` | 带误差原始点 | 代表点与式 (12) 半径 | 目标/轨迹 |
| `MatchedProxyFit` | 同一代表点 | 共同簇、球、椭球+offset | 墙标签/世界轴先验 |
| `NativeMVT` | 当前代理/点 | 连续池、AABB、候选 | 控制目标 |
| `NarrowPhase` | 机器人球、候选代理 | 净空、法向、切平面 | 人工方向 |
| `LiuQPEraseRemove` | 有序候选与支持平面 | 本周期最终约束 | 结果标签 |
| `LiuQP` | \(q_k,p_g\)、最终约束 | \(\dot q_k\) | 路点/规划路径 |
| `PostControlSweepAudit` | 已保存的 \(q_k,q_{k+1}\) 与因果代理快照 | 只读连续安全报告 | 控制回路写权限 |
| ObservabilityEvaluator | 完整日志与真值 | observed_before_risk / unobserved_hazard | 控制回路写权限 |
| `Evaluator` | 全部日志与真值 | 证书、表图 | 控制回路写权限 |

### 14.2 在线循环

```text
initialize q0, empty incremental map, one fixed final goal pg
while not terminal:
    if a depth frame is due:
        render depth AND geom-id segmentation with robot visible
        remove robot-hit pixels after rendering
        update free / occupied / unknown rays
        CenterVox-update representatives and conservative residual radii
        refit only affected common clusters; build matched spheres and ellipsoids
        build/update next native MVT snapshot and atomically publish it

    read latest completed map/MVT snapshot
    predict one-step swept AABBs for all robot certificate spheres
    obtain conservative candidates by global AABB, MVT and local AABB
    compute exact sphere or exact ellipsoid closest points
    apply LiuQP ordered erase-remove using sphere/ellipsoid support minima
    assemble the same final-goal LiuQP with hard collision constraints
    solve with warm start
    apply the QP velocity directly, or exact zero only if QP failed
    log the complete cycle

after the run:
    replay saved causal camera configurations to reconstruct map deltas
    replay brute AABB candidates and compare every online candidate count
    replay continuous proxy sweeps without modifying any saved command
```

---

## 15. 正式实现状态与证据边界

`protocol_liuqp_controller.py` 已实现 final-goal-only 软任务、硬安全行、NORMAL/NEAR/CONTACT-RECOVERY 三状态、QP 失败精确零速度回退和球/椭球共同的 ordered erase-remove。已知环境椭球核使用受保护 Newton、必要二分和 multiplier 热启动；在线含方向不确定性的代理使用式 (32a) 的受保护 Riemannian Newton、Armijo、法向热启动、中心连线重启和 \(10^{-7}\) KKT 硬门。零热启动未归一化的旧 C++ 缺陷已由单位球回归测试锁定，旧 v4 结果保留，正式证据改由修复后的 v5 给出。

正式场景具有 65 个机器人证书球和两个带实体外壳、随机械臂运动的局部深度视角。地图从 unknown 开始，射线逐帧写入 free/occupied；CenterVox 的每个代表点保留覆盖半径，代理更新通过不可变快照原子发布。C++ 原生实现包含 5 层 MVT、每次机器人球查询 135 个体素格、连续 SoA 池、inclusive float32 AABB 和 8-lane AVX2，同时提供标量核与 O(N) AABB oracle。

正式证据记录在 `formal_results/final_two_camera/formal_protocol_v3_evidence_manifest_v5.json`。静态门槛给出的球连续截面净空上界为 -5.827 mm，椭球同截面全 pair 最小余量为 +5.034 mm。在线球组运行 45 s 未到达，椭球组 12.26 s 首次到达；两组 MuJoCo 精确穿透周期均为 0，椭球最大 KKT 残差为 \(3.36\times10^{-8}\)。控制计算 p99 分别为 8.72 ms 和 16.35 ms，达到冻结的 20 ms p99 门槛，但两组各有 2 个计算周期超过 20 ms，因此不声明硬实时。

CenterVox 5/7.5/10 mm 分辨率敏感性已经完成：只有 7.5 mm 椭球组同时满足到达、可观测性、回放审计和 20 ms p99。额外 0.3 mm 严格有界深度误差的 v5 探索组数值残差合格，但代理数增至 1012、任务失败且连续 sweep 审计失败；地图重渲染在第 7 个快照还出现少量体素计数差，因此该组只作为负面工程敏感性记录，不进入主安全证据，也不能据此宣称传感器噪声鲁棒。

结论限于“椭球表示缓解了本场景中球证书造成的假闭塞，VCC-inspired 查询结构在保持候选和轨迹等价的前提下实现实时 p99”。它不证明一般非凸问题全局完备，也不对任意未观测空间给出绝对安全保证。旧长探杆、预融合多视角、路点、RRT 和 IRIS 区域链结果继续保留为历史原型，不进入正式结论。

---

## 参考文献与权威实现依据

1. C. Liu and M. Yim, “A Quadratic Programming Approach to Manipulation in Real-Time Using Modular Robots,” arXiv:2104.02755v2, 2021. 本地原文：[`liu_yim_2021.pdf`](../paper_assets/liu_yim_2021.pdf)。
2. P. Werner, T. Cohn, R. H. Jiang, et al., “Faster Algorithms for Growing Collision-Free Convex Polytopes in Robot Configuration Space,” arXiv:2410.12649v2, 2024. 本地原文：[`IRIS.pdf`](../../IRIS.pdf)；[项目主页](https://sites.google.com/view/fastiris)。
3. C. Chen and T.-T. Yeh, “VCC: Efficient Voxel-Based Collision Checking Framework for Real-Time Robotic Motion Planning,” IEEE ICRA 2026. 本地原文：[`vcc_paper.pdf`](../../vcc_paper.pdf)。
4. [MuJoCo Visualization / Cameras](https://mujoco.readthedocs.io/en/stable/programming/visualization.html)：固定相机可附着于运动刚体，渲染位姿随仿真状态更新。
5. [MuJoCo Warp API](https://mujoco.readthedocs.io/en/stable/mjwarp/api.html)：深度与逐像素 object-id/object-type segmentation 输出定义。
6. [robosuite Sensors](https://robosuite.ai/docs/modules/sensors.html) 与 [Renderers](https://robosuite.ai/docs/modules/renderers.html)：相机可附着于机器人刚体，支持 RGB、深度、geom/instance segmentation，并可建模采样率、延迟、噪声与滤波。
7. [Intel RealSense D405 官方规格](https://www.intel.com/content/www/us/en/products/sku/229218/intel-realsense-depth-camera-d405/specifications.html)：42 mm × 42 mm × 23 mm，7–50 cm 工作距离，87° × 58° 深度视场，720p 30 FPS。

---

## 文档冻结声明

本文件先定义“什么算成功”和“什么不算证据”，再允许实现和试验。后续若发现实现无法满足某项，应报告失败或以版本化修订重新冻结协议；不得静默改变场景、路径信息、感知范围或安全裕量来获得期望结果。

---

## 16. v4.2h 最终实现补充（取代第 15 节旧 v5 主结论）

最终在线代理不再使用第 15 节所述的 \(Q+U+\delta\) 方向支持近似。每个 12.5 mm CenterVox 的累计带误差 support interval 沿数据确定的两个切向坐标做 2×2 子盒分片；每个子盒只构造一次

\[
Q=R\operatorname{diag}(a_n^2,a_{t_1}^2,a_{t_2}^2)R^\top,\qquad
a_n\le 1.10\,\bar h_n,
\]

其中 \(w_n=1/1.10^2\)、\(w_{t_1}=w_{t_2}=(1-w_n)/2\)、\(a_i=\bar h_i/\sqrt{w_i}\)。发布后固定 \(U=0\)、\(\delta=0\)，6 mm 安全余量不进入 \(Q\)。

机器人球 \(B(p,r)\) 与障碍椭球 \(E(c,Q)\) 的在线净空为

\[
d(p,E)-r-d_{\mathrm{safe}},
\qquad
d(p,E)=\min_{x:\,(x-c)^\top Q^{-1}(x-c)\le1}\|p-x\|_2 .
\]

最近点乘子方程使用上一周期热启动、受保护 Newton，无法保持括区或残差门时才二分；由最近点表面法向构造 LiuQP 硬行。不存在 `ellipsoid-fast` 中心线支持替代。球型使用同中心、同数量代理和 \(r_j=\sqrt{\lambda_{\max}(Q_j)}\)。

VCC-inspired 在线流水线为：CenterVox 证书发布；C++ 多级体素表的局部 27 格查询；SoA AABB；AVX2 八宽球--AABB；距离下界排序；原生精确窄阶段；IRIS-inspired ordered erase-remove。粗阶段代理 AABB 和查询半径各加 5 微米只增假阳性，float64 oracle 与事后重放均证明无漏候选。

OSQP 只做数值工程优化：低行数使用 rho=0.1 自适应模式，高行数使用 rho=0.01 固定模式；模式改变时重建工作区。上一周期对偶变量按 joint/workspace/collision/recovery 语义行身份重映射，保留未变行、将新行置零。QP 矩阵与上下界哈希不因该策略改变。

最终正式结果为：匹配球 60 s 后误差 318.914 mm，精确椭球 58.18 s 完成严格小于 1 mm 连续 50 周期且 60 s 末误差 0.970 mm；控制器 p99 分别 11.41 ms 和 13.49 ms。实际球型因果快照的必经截面连续净空上界为 \(-8.586\ \mu m\)，而同一快照匹配椭球严格开口为 6.288 mm。完整证据和使用方法见 `V4_2H_FORMAL_RESULTS_AND_USAGE.md`。
