# 第二章 以原始 LiuQP 推导为母版的椭球型 LiuQP 数学推导

## 2.1 推导范围与符号方向

本章以《LiuQP Mathematical Derivation and Theoretical Guide》为母版。母版把 LiuQP 分成图结构运动学、任务反馈、关节与工作空间约束、球树障碍物、分离平面、冗余球删除、综合 QP 以及 NORMAL/NEAR/CONTACT 切换。把环境球改为环境椭球时，机械臂的运动学、末端速度、模块球心速度、任务反馈和关节边界并不改变；发生变化的是障碍物表面最近点、分离法向、表面间隙、冗余代理判定以及由这些量生成的 QP 行。因此，本章保留母版公式（1）—（22）的含义，从母版补充式 A—H 和公式（23）开始进行椭球推广。

为与母版的公式 B、（23）和（27）保持一致，本章令 \(\tilde s_{ij}\) 表示从第 \(i\) 个机器人证书指向第 \(j\) 个障碍物证书的单位法向。第一章使用的 \(n_{ij}\) 从障碍物指向机器人，二者满足

\[
n_{ij}=-\tilde s_{ij}.
\tag{E-0}
\]

这个符号选择只改变公式两侧的正负号，不改变几何含义。当前代码中的 `normal_to_obstacle` 与本章 \(\tilde s_{ij}\) 同向，因此本章公式可以直接映射到实现。机器人仍采用球证书；环境采用椭球或带方向不确定性的椭球 Minkowski 和。机械臂本体改为椭球会引入转动导致的支持半径导数，不属于当前主对照。

## 2.2 直接复用的机械臂运动学与模块点速度

母版先从模块图得到世界坐标系到任务坐标系的运动学链。对模块本体坐标系 \(M\) 与连接坐标系 \(C\)，关节旋量为 \(\xi_{\theta_\ell}\)，乘积指数形式为

\[
g_{MC}(\Theta^C)
=\prod_{\ell}\exp\!\left(\hat\xi_{\theta_\ell}\theta_\ell^C\right)
g_{MC}(0).
\tag{1}
\]

沿世界坐标系 \(W\) 到任务坐标系 \(F\) 的图路径连乘可得 \(g_{WF}\)，任务坐标系原点在世界坐标系中的位置为

\[
p_F^W=g_{WF}
\begin{bmatrix}0&0&0&1\end{bmatrix}^{T}.
\tag{2}
\]

对该链上的全部关节求导，空间速度与空间雅可比满足

\[
\hat V_{WF}^{s}
=\sum_i\sum_j
\frac{\partial g_{WF}}{\partial\theta_{ij}}
g_{WF}^{-1}\dot\theta_{ij},
\tag{3}
\]

\[
V_{WF}^{s}=J_{WF}^{s}\dot\Theta^{WF},
\qquad
J_{WF}^{s}=\begin{bmatrix}J_1&J_2&\cdots&J_N\end{bmatrix}.
\tag{4--6}
\]

其中每个模块贡献的雅可比块可以写为

\[
J_i=
\begin{bmatrix}
\left(\frac{\partial g}{\partial\theta_{i1}}g^{-1}\right)^{\vee}
&\cdots&
\left(\frac{\partial g}{\partial\theta_{iN_i}}g^{-1}\right)^{\vee}
\end{bmatrix}
=\begin{bmatrix}\xi'_{i1}&\cdots&\xi'_{iN_i}\end{bmatrix}.
\tag{7--8}
\]

空间旋量不能直接作为碰撞球心的线速度。母版利用齐次点坐标把六维空间旋量转换为三维点速度：

\[
v_F^{s}=\hat V_{WF}^{s}p_F^W
=\widehat{\left(J_{WF}^{s}\dot\Theta^{WF}\right)}p_F^W.
\tag{9}
\]

在常用向量表示下，空间旋量 \([v^T,\omega^T]^T\) 在点 \(p\) 处产生 \(v+\omega\times p\)。这一运算把 \(6\times n\) 的空间雅可比变成控制与碰撞约束实际使用的 \(3\times n\) 点雅可比。环境由球改为椭球不会改变该步骤。

碰撞约束需要每个机器人模块或每个机器人证书球自己的点雅可比，而不是只使用末端雅可比。对第 \(i\) 个模块，从 \(W\) 到 \(M_i\) 的子链雅可比为

\[
J_{M_i}^{s}
=\begin{bmatrix}\xi'_{11}&\xi'_{12}&\cdots&
\xi'_{\bar i\bar j_i}\end{bmatrix},
\tag{10}
\]

其球心速度为

\[
v_{M_i}^{s}
=\widehat{\left(J_{M_i}^{s}\dot\Theta^{WM_i}\right)}p_{M_i}^{W}.
\tag{11}
\]

为让所有约束共享同一个全局决策向量，需要把该模块下游、不会影响它的关节列补零：

\[
J_{M_i}^{s}
=\begin{bmatrix}
\xi'_{11}&\cdots&\xi'_{\bar i\bar j_i}
&0_{6\times1}&\cdots&0_{6\times1}
\end{bmatrix},
\tag{12}
\]

\[
v_{M_i}^{s}
=\widehat{\left(J_{M_i}^{s}\dot\Theta^{WF}\right)}p_{M_i}^{W}
=J_{p,i}(q)\dot q.
\tag{13}
\]

后续椭球约束中出现的 \(J_{p,i}\) 正是式（13）的三维点雅可比。椭球最近点只负责给出当前法向和净空，不替代模块运动学，也不允许把所有机器人球错误地绑定到 \(J_{ee}\)。

## 2.3 直接复用的末端反馈、任务软化和关节边界

设期望任务位置为 \(\tilde p_F(t)\)，期望速度为 \(\tilde v_F(t)\)，误差为 \(e=\tilde p_F-p_F\)。母版规定一阶稳定误差动力学

\[
\dot e+Ke=0,
\qquad K\succ0.
\tag{14}
\]

把 \(\dot e=\tilde v_F-v_F\) 代入可得

\[
\tilde v_F^{s}-v_F^{s}
+K(\tilde p_F-p_F)=0,
\tag{15}
\]

\[
\widehat{\left(J_{WF}^{s}\dot\Theta^{WF}\right)}p_F
=\tilde v_F^{s}+K(\tilde p_F-p_F).
\tag{16}
\]

将一个或多个任务坐标系的点雅可比堆叠后，得到

\[
J\dot q=\tilde V+K(\tilde P-P)=b.
\tag{17}
\]

硬任务形式对应最小范数问题

\[
\begin{aligned}
\min_{\dot q}\quad &\frac12\dot q^T\dot q,\\
\mathrm{s.t.}\quad &J\dot q=b.
\end{aligned}
\tag{18}
\]

一步 Euler 积分 \(q^+=q+\Delta t\dot q\) 把关节位置界转换为

\[
\frac{q_{\min}-q}{\Delta t}
\le\dot q\le
\frac{q_{\max}-q}{\Delta t},
\tag{19}
\]

同时保留执行器速度界

\[
\dot q_{\min}\le\dot q\le\dot q_{\max}.
\tag{20}
\]

当式（17）与碰撞或硬件边界冲突时，硬等式会导致 QP 不可行。母版随后把任务跟踪移入二次目标，这一处理在椭球版中原样保留。换言之，椭球化只改变碰撞几何，不改变末端追踪律，也不能通过调整目标或增加中间路点帮助机器人进入抽屉。

工作空间平面约束同样不依赖环境代理形状。若 \(\hat s_{if}\) 从机器人模块球心指向工作空间边界面，中心到平面的距离为 \(d_{if}\)，母版公式（21）—（22）为

\[
{v_{M_i}^{s}}^T\hat s_{if}\le d_{if}-r_i.
\tag{21--22}
\]

母版也指出该式左侧是速度、右侧是距离。采用明确控制周期后，应写为

\[
\hat s_{if}^{T}J_{p,i}(q)\dot q
\le\frac{d_{if}-r_i}{\Delta t}.
\tag{22E}
\]

## 2.4 从障碍球改为连续障碍椭球证书

第 \(i\) 个机器人证书仍为闭球

\[
\mathcal B_i(q)=B\bigl(p_i(q),r_i\bigr).
\tag{E-A1}
\]

第 \(j\) 个障碍核心椭球定义为

\[
\mathcal E_j=E(o_j,Q_j)
=\left\{x:(x-o_j)^TQ_j^{-1}(x-o_j)\le1\right\},
\qquad Q_j\succ0.
\tag{E-A2}
\]

若 \(Q_j=R_j\operatorname{diag}(a_{j1}^2,a_{j2}^2,a_{j3}^2)R_j^T\)，则 \(R_j\) 给出主轴方向，\(a_{j\ell}\) 给出三个半轴。与半径在所有方向相同的球不同，椭球在单位方向 \(s\) 上的支持半径为

\[
\rho_{Q_j}(s)=\sqrt{s^TQ_js}.
\tag{E-A3}
\]

在线点云证书还可以保留方向不确定性 \(U_j\succeq0\) 和各向同性覆盖偏置 \(\delta_j\ge0\)：

\[
\mathcal K_j
=E(o_j,Q_j)\oplus E(0,U_j)\oplus B(0,\delta_j).
\tag{E-A4}
\]

这里 \(Q_j\)、\(U_j\) 与 \(\delta_j\) 均属于环境代理自身的覆盖定义；控制安全余量 \(d_{\mathrm{safe}}\) 不得再次并入这些量。主实验若发布的是单一表面核心椭球，则令 \(U_j=0\)；若覆盖偏置已经进入椭球构造，则相应令 \(\delta_j=0\)。每个分支都必须由冻结配置明确，不能在公式中重复膨胀。

球—椭球的几何安全条件为

\[
\operatorname{dist}\bigl(p_i(q),\mathcal K_j\bigr)-r_i
\ge d_{\mathrm{safe}}.
\tag{E-A5}
\]

式（E-A5）仍然是关节位置的非线性函数。椭球型 LiuQP 的任务不是把它直接塞入 QP，而是在当前构型求出真实最近分离平面，再把一步运动限制写成 \(\dot q\) 的线性不等式。这与母版从球—球距离转向切平面的逻辑一致。

<details>
<summary>1. 为什么椭球可以写成式（E-A2）</summary>
定义
\[
\mathcal E_j=
\left\{
x:(x-o_j)^TQ_j^{-1}(x-o_j)\le 1
\right\},
\qquad Q_j\succ0.
\]由于 \(Q_j\) 对称正定，可以分解为
\[
Q_j=
R_j
\operatorname{diag}
(a_{j1}^2,a_{j2}^2,a_{j3}^2)
R_j^T.
\]令
\[
A_j=
R_j\operatorname{diag}(a_{j1},a_{j2},a_{j3}),
\qquad Q_j=A_jA_j^T.
\]那么椭球也可以参数化为
\[
x=o_j+A_ju,
\qquad \|u\|_2\le1.
\]把它代入二次型：
\[
\begin{aligned}
(x-o_j)^TQ_j^{-1}(x-o_j)
&=(A_ju)^T(A_jA_j^T)^{-1}(A_ju)\\
&=u^Tu\\
&\le1.
\end{aligned}
\]因此，式（E-A2）其实就是“单位球经过旋转、三个方向分别缩放，再平移到 \(o_j\)”的结果：
\[
B(0,1)
\xrightarrow{\operatorname{diag}(a_{j1},a_{j2},a_{j3})}
\text{轴对齐椭球}
\xrightarrow{R_j}
\text{旋转椭球}
\xrightarrow{o_j}
\mathcal E_j.
\]所以：
- \(o_j\) 是椭球中心；
- \(R_j\) 的三列是三条主轴方向；
- \(a_{j1},a_{j2},a_{j3}\) 是三个半轴长度；
- \(Q_j\) 中保存的是半轴长度的平方和方向信息。
继续补充为什么任意对称正定矩阵都能写成
\[
Q=R\operatorname{diag}(a_1^2,a_2^2,a_3^2)R^T,
\]以及这个形式为什么对应椭球。可以从“单位球如何变成椭球”推出来。

#### (1). diag 表示什么
\[
\operatorname{diag}(a_1^2,a_2^2,a_3^2)
\]表示对角矩阵
\[
\begin{bmatrix}
a_1^2&0&0\\
0&a_2^2&0\\
0&0&a_3^2
\end{bmatrix}.
\]它不会混合不同坐标轴，只会分别缩放三个坐标方向。
例如
\[
D=
\operatorname{diag}(a_1,a_2,a_3)
\]作用在向量
\[
u=
\begin{bmatrix}
u_1\\u_2\\u_3
\end{bmatrix}
\]上得到
\[
Du=
\begin{bmatrix}
a_1u_1\\
a_2u_2\\
a_3u_3
\end{bmatrix}.
\]所以它把 \(x,y,z\) 三个方向分别放大 \(a_1,a_2,a_3\) 倍。
#### (2). 从单位球开始构造椭球
三维单位球可以写成
\[
\mathcal B=
\{u:u^Tu\le1\}.
\]展开就是
\[
u_1^2+u_2^2+u_3^2\le1.
\]现在用
\[
D=\operatorname{diag}(a_1,a_2,a_3)
\]对单位球进行缩放，令
\[
y=Du.
\]于是
\[
y_1=a_1u_1,\qquad
y_2=a_2u_2,\qquad
y_3=a_3u_3.
\]反过来，
\[
u_1=\frac{y_1}{a_1},\qquad
u_2=\frac{y_2}{a_2},\qquad
u_3=\frac{y_3}{a_3}.
\]代回单位球公式：
\[
\frac{y_1^2}{a_1^2}+
\frac{y_2^2}{a_2^2}+
\frac{y_3^2}{a_3^2}
\le1.
\]这就是一个轴对齐椭球。其中 \(a_1,a_2,a_3\) 正好是三个半轴长度。
矩阵形式为
\[
y^T
\begin{bmatrix}
1/a_1^2&0&0\\
0&1/a_2^2&0\\
0&0&1/a_3^2
\end{bmatrix}
y
\le1.
\]也就是
\[
y^TD^{-2}y\le1.
\]

#### (3). 再旋转这个椭球

轴对齐椭球的主轴固定在世界坐标系的 \(x,y,z\) 方向。为了表示任意朝向，需要一个旋转矩阵 \(R\)。
令
\[
x-o=Ry,
\]其中 \(o\) 是椭球中心。因为旋转矩阵满足
\[
R^TR=RR^T=I,
\]所以
\[
y=R^T(x-o).
\]将它代入轴对齐椭球方程：
\[
y^TD^{-2}y\le1,
\]得到
\[
(x-o)^TRD^{-2}R^T(x-o)\le1.
\]因为
\[
(RD^2R^T)^{-1}=
RD^{-2}R^T,
\]定义
\[
Q=RD^2R^T,
\]便得到
\[
(x-o)^TQ^{-1}(x-o)\le1.
\]又因为
\[
D^2=
\operatorname{diag}(a_1^2,a_2^2,a_3^2),
\]所以
\[
\boxed{
Q=R\operatorname{diag}(a_1^2,a_2^2,a_3^2)R^T
}
\]
不是凭空规定出来的，而是由以下几步自然得到的：
\[
\text{单位球}
\xrightarrow{D}
\text{沿三个方向缩放}
\xrightarrow{R}
\text{旋转}
\xrightarrow{o}
\text{平移}.
\]完整参数方程就是
\[
\boxed{
x=o+RDu,\qquad \|u\|\le1.
}
\]
#### (4). 为什么任意对称正定矩阵都能这样分解

这里使用的是实对称矩阵的谱定理。
对于任意实对称矩阵 \(Q=Q^T\)，都存在一组单位正交特征向量
\[r_1,r_2,r_3\]和对应特征值
\[\lambda_1,\lambda_2,\lambda_3,\]满足
\[Qr_i=\lambda_i r_i.\]将三个特征向量作为矩阵的列：
\[
R=\begin{bmatrix}
r_1&r_2&r_3
\end{bmatrix}.
\]由于这些特征向量彼此正交且长度为 1，
\[
R^TR=I.
\]将三个特征值放入对角矩阵：
\[
\Lambda=\operatorname{diag}(\lambda_1,\lambda_2,\lambda_3).
\]三个特征向量方程可以合并写成
\[
QR=R\Lambda.
\]右乘 \(R^T\)，得到
\[
QRR^T=R\Lambda R^T.
\]因为 \(RR^T=I\)，所以
\[
\boxed{
Q=R\Lambda R^T.
}
\]这就是实对称矩阵的特征值分解。

</details>

<details>
<summary>2. 为什么支持半径是 \(\sqrt{s^TQ_js}\)</summary>
这里的“支持半径”不是从中心沿射线 \(s\) 到椭球表面的距离，而是椭球在方向 \(s\) 上的最大投影。
对单位方向 \(\|s\|_2=1\)，定义
\[
\rho_{Q_j}(s)=
\max_{x\in\mathcal E_j}s^T(x-o_j).
\]利用 \(x=o_j+A_ju\)，得到
\[
\begin{aligned}
\rho_{Q_j}(s)&=
\max_{\|u\|\le1}s^TA_ju\\&=
\max_{\|u\|\le1}(A_j^Ts)^Tu.
\end{aligned}
\]根据 Cauchy–Schwarz 不等式，
\[
(A_j^Ts)^Tu
\le
\|A_j^Ts\|_2\|u\|_2
\le
\|A_j^Ts\|_2.
\]当
\[
u^*=\frac{A_j^Ts}{\|A_j^Ts\|}
\]时取到等号。因此
\[
\begin{aligned}
\rho_{Q_j}(s)
&=\|A_j^Ts\|_2\\
&=\sqrt{s^TA_jA_j^Ts}\\
&=\sqrt{s^TQ_js}.
\end{aligned}
\]这就得到式（E-A3）：
\[
\boxed{
\rho_{Q_j}(s)=\sqrt{s^TQ_js}
}
\]对应的支持点为
\[
x_j^+(s)=
o_j+\frac{Q_js}{\sqrt{s^TQ_js}},
\]而朝向 \(-s\) 的支持点为
\[
x_j^-(s)=
o_j-\frac{Q_js}{\sqrt{s^TQ_js}}.
\]椭球—机器人分离平面需要的是支持点，所以必须使用支持半径，而不是射线半径
\[
r_{\mathrm{ray}}(s)=
\frac{1}{\sqrt{s^TQ_j^{-1}s}}.
\]这两个量对一般椭球并不相等。只有球或者某些主轴方向上才可能一致。
</details>


<details>
<summary>3. 为什么在线证书写成 Minkowski 和</summary>
核心椭球是从局部点云拟合出来的几何主体：
\[
E(o_j,Q_j).
\]但是，拟合椭球本身未必能够覆盖所有真实障碍点。误差可能包含两部分。
第一部分是具有明显方向性的误差，例如：
- 某个方向上的点云稀疏；
- 深度方向的不确定性大于切向方向；
- CenterVox 代表点对原始点集的方向性偏差；
- 椭球拟合后某一主轴方向仍存在残差。
用
\[
E(0,U_j)
\]描述这种各向异性误差。
第二部分是不适合指定方向的小量误差，例如：
- 深度相机随机噪声；
- 体素量化误差；
- 浮点误差；
- 时间同步误差；
- 尚未归入方向矩阵的统一保护量。
用半径为 \(\delta_j\) 的球
\[
B(0,\delta_j)
\]描述。
于是完整执行证书写为
\[
\boxed{
\mathcal K_j
=
E(o_j,Q_j)
\oplus E(0,U_j)
\oplus B(0,\delta_j)
}
\]其中 Minkowski 和定义为
\[
A\oplus B
=
\{a+b:a\in A,\ b\in B\}.
\]所以任意 \(x\in\mathcal K_j\) 都能写成
\[
x
=
o_j+Q_j^{1/2}u
+U_j^{1/2}v
+\delta_jw,
\]其中
\[
\|u\|\le1,\qquad
\|v\|\le1,\qquad
\|w\|\le1.
\]它表达的是：
\[
\text{执行证书}
=
\text{核心椭球}+
\text{各向异性误差}+
\text{各向同性误差}.
\]
这里不是三个重复障碍物，也不是“大椭球套小椭球”产生三个 QP 约束。它们共同定义一个复合凸证书 \(\mathcal K_j\)，最终只针对这个证书生成碰撞约束。
</details>

<details>
<summary>4. 为什么这些支持半径可以直接相加</summary>
支持函数具有一个关键性质：
\[
h_{A\oplus B}(s)=h_A(s)+h_B(s).
\]因此
\[
\begin{aligned}
h_{\mathcal K_j}(s)&=
s^To_j
+\sqrt{s^TQ_js}
+\sqrt{s^TU_js}
+\delta_j\|s\|.
\end{aligned}
\]当 \(\|s\|=1\) 时，
\[
\boxed{
h_{\mathcal K_j}(s)=
s^To_j
+\sqrt{s^TQ_js}
+\sqrt{s^TU_js}
+\delta_j
}
\]所以，从机器人指向障碍物的方向为 \(s\) 时，障碍物朝向机器人的最小投影位置为
\[
s^To_j-\sqrt{s^TQ_js}-\sqrt{s^TU_js}-\delta_j.
\]如果机器人证书也是椭球 \(E(p_i,Q_i^R)\)，机器人朝向障碍物的最大投影为
\[
s^Tp_i+\sqrt{s^TQ_i^Rs}.
\]二者之差正好是
\[
\phi_{ij}(s)=
s^T(o_j-p_i)
-\sqrt{s^TQ_i^Rs}
-\sqrt{s^TQ_js}
-\sqrt{s^TU_js}
-\delta_j.
\]这就是后续分离间隙公式的来源。它不是经验拼接，而是由 Minkowski 和的支持函数严格推导出来的。
</details>

<details>
<summary>5. \(U_j\succeq0\) 为什么允许为半正定</summary>
核心椭球要求
\[
Q_j\succ0,
\]因为它需要是具有三个非零半轴的实体椭球，并且式（E-A2）中要使用 \(Q_j^{-1}\)。
但不确定性可能只存在于一个或两个方向。例如，只有相机深度方向存在明显误差。此时 \(U_j\) 可以是半正定矩阵：
\[
U_j\succeq0.
\]更严格地说，应该把它定义为
\[
E(0,U_j)=
\left\{
U_j^{1/2}v:\|v\|\le1
\right\},
\]而不是强行使用 \(U_j^{-1}\)。如果 \(U_j=0\)，那么
\[
E(0,U_j)=\{0\},
\qquad
\sqrt{s^TU_js}=0.
\]实现时直接省略这一项，不能计算
\[
\frac{U_js}{\sqrt{s^TU_js}},
\]否则会出现 \(0/0\)。
</details>

<details>
<summary>6. 什么时候它才是真正的障碍物外包证书</summary>
公式的凸几何推导是正确的，但“它完整包住真实障碍物”还需要一个额外条件：
\[
\mathcal O_j^{\mathrm{true}}
\subseteq
\mathcal K_j.
\]也就是分配给第 \(j\) 个代理的全部真实障碍点和未采样表面，都必须落在 \(\mathcal K_j\) 内部。
如果核心椭球只是普通最小二乘拟合，而 \(U_j,\delta_j\) 又是随意设置的，那么式（E-A4）虽然在数学上仍定义了一个凸集，却不能称为经过证明的覆盖证书。实验中必须验证：
\[
\max_{x\in\mathcal P_j}
\operatorname{dist}
\bigl(x,E(o_j,Q_j)\oplus E(0,U_j)\bigr)
\le\delta_j,
\]或者使用等价的支持方向、连续截面与体素覆盖审计。
因此，正确理解是：
\[
\boxed{
\text{式（E-A4）给出正确的证书结构；
覆盖参数是否正确，需要由点云残差和覆盖审计证明。}
}
\]另外，\(Q_j\)、\(U_j\)、\(\delta_j\) 与控制安全距离 \(d_{\mathrm{safe}}\) 是不同层次。前三者保证环境几何被覆盖，\(d_{\mathrm{safe}}\) 是机器人额外保持的控制距离，不能把同一余量重复加入两次。
</details>

---

## 2.5 机器人球—单个障碍椭球的真实最近点

先考虑 \(U_j=0\) 的核心情形。给定机器人球心 \(p_i\)，椭球表面上离它最近的点由

\[
x_{ij}^{*}
=\arg\min_x\frac12\|x-p_i\|_2^2,
\quad
\mathrm{s.t.}\quad
(x-o_j)^TQ_j^{-1}(x-o_j)=1
\tag{E-B1}
\]

确定。对位于椭球外部的 \(p_i\)，约束在最优点处激活。定义拉格朗日函数

\[
\mathcal L(x,\lambda)
=\frac12\|x-p_i\|_2^2
+\frac{\lambda}{2}
\left((x-o_j)^TQ_j^{-1}(x-o_j)-1\right).
\tag{E-B2}
\]

一阶驻点条件为

\[
x-p_i+\lambda Q_j^{-1}(x-o_j)=0.
\tag{E-B3}
\]

令 \(Q_j=R_j\operatorname{diag}(a_1^2,a_2^2,a_3^2)R_j^T\)，并定义局部坐标

\[
y=R_j^T(p_i-o_j),
\qquad z=R_j^T(x-o_j).
\tag{E-B4}
\]

式（E-B3）在每个主轴方向上变成

\[
z_\ell-y_\ell+\frac{\lambda}{a_\ell^2}z_\ell=0,
\qquad
z_\ell=\frac{a_\ell^2y_\ell}{a_\ell^2+\lambda}.
\tag{E-B5}
\]

把式（E-B5）代回椭球边界约束，可把三维最近点问题化成唯一的一维根：

\[
F(\lambda)
=\sum_{\ell=1}^{3}
\frac{a_\ell^2y_\ell^2}{(a_\ell^2+\lambda)^2}-1=0,
\qquad \lambda\ge0,
\tag{E-B6}
\]

\[
F'(\lambda)
=-2\sum_{\ell=1}^{3}
\frac{a_\ell^2y_\ell^2}{(a_\ell^2+\lambda)^3}<0.
\tag{E-B7}
\]

当 \(p_i\) 位于椭球外部时，\(F(0)>0\)，而 \(F(\lambda)\to-1\) 随 \(\lambda\to\infty\)。因此正根唯一。实现先维护满足 \(F(\lambda_L)\ge0\)、\(F(\lambda_U)\le0\) 的括区，再执行

\[
\lambda_{m+1}
=\lambda_m-\frac{F(\lambda_m)}{F'(\lambda_m)}.
\tag{E-B8}
\]

若 Newton 候选不是有限数、离开括区或导数条件退化，就改用 \((\lambda_L+\lambda_U)/2\)；Newton 预算结束后仍未达到残差门，继续二分至收敛。上一控制周期中同一稳定 robot—proxy ID 的乘子可作为本周期初值，这就是乘子热启动。二分不是每个 pair 固定执行几十次，而是保护 Newton 无法满足括区与残差要求时的可靠回退。

求得 \(\lambda^*\) 后，最近点为

\[
x_{ij}^{*}
=o_j+R_j
\begin{bmatrix}
\dfrac{a_1^2y_1}{a_1^2+\lambda^*}\\[3pt]
\dfrac{a_2^2y_2}{a_2^2+\lambda^*}\\[3pt]
\dfrac{a_3^2y_3}{a_3^2+\lambda^*}
\end{bmatrix}.
\tag{E-B9}
\]

为延续母版方向，定义从机器人球心指向该最近点的单位向量

\[
\tilde s_{ij}
=\frac{x_{ij}^{*}-p_i}{\|x_{ij}^{*}-p_i\|_2}.
\tag{E-B10}
\]

椭球在 \(-\tilde s_{ij}\) 方向上的支持点正是 \(x_{ij}^{*}\)，所以

\[
x_{ij}^{*}
=o_j-\frac{Q_j\tilde s_{ij}}
{\sqrt{\tilde s_{ij}^TQ_j\tilde s_{ij}}}.
\tag{E-B11}
\]

式（E-B11）说明中心连线一般不等于真实最近法向。只有球或机器人球心恰好位于椭球主轴等特殊情形，\((o_j-p_i)/\|o_j-p_i\|\) 才与 \(\tilde s_{ij}\) 重合。把中心连线直接代入支持函数会改变几何问题，不是本研究接受的快速近似。

## 2.6 椭球切平面、原始间隙与速度级线性约束

若代理包含各向同性偏置 \(\delta_j\)，核心最近点沿机器人方向外移后的代理表面点为

\[
o'_{ij}=x_{ij}^{*}-\delta_j\tilde s_{ij}.
\tag{E-C}
\]

对应分离平面为

\[
\mathcal P_{ij}^{E}
=\left\{x:\tilde s_{ij}^T(x-o'_{ij})=0\right\}.
\tag{E-D}
\]

机器人位于平面的负侧，障碍椭球位于正侧。第 \(i\) 个机器人球完全位于安全侧的条件是

\[
\tilde s_{ij}^T(p_i-o'_{ij})+r_i\le0.
\tag{E-E}
\]

尚未减去控制安全余量的证书表面间隙定义为

\[
c_{ij}
=\|x_{ij}^{*}-p_i\|_2-r_i-\delta_j.
\tag{E-F1}
\]

利用式（E-B11），同一间隙也可写成支持函数形式

\[
c_{ij}
=\tilde s_{ij}^T(o_j-p_i)
-\sqrt{\tilde s_{ij}^TQ_j\tilde s_{ij}}
-r_i-\delta_j.
\tag{E-F2}
\]

定义硬安全间隙

\[
h_{ij}=c_{ij}-d_{\mathrm{safe}}.
\tag{E-F3}
\]

在当前控制周期，\(p_i\)、\(J_{p,i}\)、\(o'_{ij}\) 和 \(\tilde s_{ij}\) 均被冻结。一步后机器人球心近似为 \(p_i^+=p_i+\Delta tJ_{p,i}\dot q\)。要求带安全余量的机器人球不越过切平面：

\[
\tilde s_{ij}^T(p_i^+-o'_{ij})
+r_i+d_{\mathrm{safe}}\le0.
\tag{E-F4}
\]

代入 \(p_i^+\) 并整理得到椭球版母版公式（23）：

\[
\tilde s_{ij}^TJ_{p,i}(q)\dot q
\le\frac{h_{ij}}{\Delta t}.
\tag{23E}
\]

这一行关于当前 QP 决策量 \(\dot q\) 是仿射的。椭球最近点求解发生在 QP 组装之前；Newton 和二分不会进入 QP 变量。由于 \(\tilde s_{ij}\) 是当前距离问题的最优法向，包络定理给出

\[
\dot c_{ij}
=-\tilde s_{ij}^{T}J_{p,i}(q)\dot q.
\tag{E-F5}
\]

不需要在式（E-F5）中继续对 \(\tilde s_{ij}\) 或 \(\lambda^*\) 求导。式（23E）也可由 \(c_{ij}^{+}\approx c_{ij}+\Delta t\dot c_{ij}\ge d_{\mathrm{safe}}\) 直接得到。

## 2.7 含机器人形状、障碍椭球和方向不确定性的统一支持和

一维乘子方程适用于机器人球—单个障碍核心椭球。若需要显式保留机器人形状矩阵 \(Q_i^R\)、障碍形状矩阵 \(Q_j^O\)、方向不确定性 \(U_j\) 和偏置 \(\delta_j\)，可直接在单位球面求两个凸证书的最大分离量。定义

\[
\phi_{ij}(s)
=s^T(o_j-p_i)
-\sqrt{s^TQ_i^Rs}
-\sqrt{s^TQ_j^Os}
-\sqrt{s^TU_js}
-\delta_j,
\qquad \|s\|_2=1.
\tag{E-G1}
\]

对不相交的凸证书，欧氏表面间隙为

\[
c_{ij}=\max_{\|s\|_2=1}\phi_{ij}(s),
\qquad
h_{ij}=c_{ij}-d_{\mathrm{safe}}.
\tag{E-G2}
\]

这就是此前讨论的
\[s^T(o-p)-\sqrt{s^TQ_Rs}-\sqrt{s^TQ_Os}-\sqrt{s^TUs}-\delta-d_{\mathrm{safe}}\]
的完整定义。该表达式只有在最优法向 \(s^*\) 处才是两个证书的真实最短分离量；任取一个方向代入只能得到该方向的投影间隙。若两证书已经相交，式（E-G2）的最大值不大于零，可用于 CONTACT 分类，但其绝对值不应未经证明地解释成一般凸体的最小平移穿透深度。

<details>
<summary>补充推导</summary>
2.7 的核心换了一个视角：
\[
\boxed{\text{2.6 是“最近点 KKT”}}
\]\[
\boxed{\text{2.7 是“支持函数 / 分离超平面”}}
\]它不是再直接找椭球上的最近点，而是找一个单位方向 \(s\)，使两个凸体在这个方向上的投影间隙最大。
先提醒一个很关键的排版点：你贴出来的式子里很多平方根像是丢了。若 \(Q\) 是椭球形状矩阵，则支持半径应为
\[
\rho_Q(s)=\sqrt{s^TQs},
\]不是 \(s^TQs\)。否则 E-G3、E-G6、E-G8 都对不上。正确理解应是：
\[
\phi_{ij}(s)
=
s^T(o_j-p_i)
-
\sqrt{s^TQ_i^R s}
-
\sqrt{s^TQ_j^O s}
-
\sqrt{s^TU_j s}
-
\delta_j ,
\qquad \|s\|=1.
\]1. 先定义两个凸证书
机器人形状证书可以看成椭球：
\[
\mathcal R_i
=
\left\{
p_i+v_R
\mid
v_R^T(Q_i^R)^{-1}v_R\le 1
\right\}.
\]障碍物证书可以看成核心椭球再加方向不确定性和偏置膨胀：
\[
\mathcal O_j
=
o_j
\oplus
\mathcal E(Q_j^O)
\oplus
\mathcal E(U_j)
\oplus
\mathcal B(\delta_j).
\]直观上：
\[
\text{障碍证书}
=
\text{障碍核心椭球}
+
\text{方向不确定性膨胀}
+
\text{偏置安全膨胀}.
\]2. 一个方向上的投影间隙
给定单位方向 \(s\)，机器人朝障碍物方向的最远投影是
\[
\max_{x_R\in\mathcal R_i}s^Tx_R.
\]障碍物朝机器人方向的最近投影是
\[
\min_{x_O\in\mathcal O_j}s^Tx_O.
\]所以这个方向上的间隙是
\[
\phi_{ij}(s)
=
\min_{x_O\in\mathcal O_j}s^Tx_O
-
\max_{x_R\in\mathcal R_i}s^Tx_R.
\]如果这个值为正，说明存在两条垂直于 \(s\) 的平行平面把二者分开：
\[
\underbrace{s^Tx_R}_{\text{机器人最大投影}}
\quad < \quad
\underbrace{s^Tx_O}_{\text{障碍物最小投影}}.
\]因为 \(\|s\|=1\)，所以投影差值就是欧氏距离单位下的间隙。
3. 椭球支持函数怎么来
对一个椭球
\[
\mathcal E_Q(p)
=
\{p+v\mid v^TQ^{-1}v\le 1\},
\]要求方向 \(s\) 上的最大投影：
\[
\max_v s^T(p+v),
\qquad
v^TQ^{-1}v\le 1.
\]去掉常数 \(s^Tp\)，只看
\[
\max_v s^Tv.
\]拉格朗日函数：
\[
L(v,\mu)
=
s^Tv-\mu(v^TQ^{-1}v-1).
\]一阶条件：
\[
\frac{\partial L}{\partial v}
=
s-2\mu Q^{-1}v=0.
\]所以
\[
v=\frac{1}{2\mu}Qs.
\]代回约束：
\[
v^TQ^{-1}v=1,
\]得到
\[
\frac{1}{(2\mu)^2}s^TQs=1.
\]因此
\[
2\mu=\sqrt{s^TQs}.
\]于是支持点是
\[
x^+(s)
=
p+\frac{Qs}{\sqrt{s^TQs}}.
\]最大投影值是
\[
s^Tx^+(s)
=
s^Tp+\sqrt{s^TQs}.
\]4. 所以机器人支持点是 E-G3
机器人朝障碍物方向的支持点：
\[
x_R(s)
=
p_i+
\frac{Q_i^R s}{\sqrt{s^TQ_i^Rs}}.
\]对应最大投影：
\[
s^Tx_R(s)
=
s^Tp_i+\sqrt{s^TQ_i^Rs}.
\]如果机器人是球，令
\[
Q_i^R=r_i^2I,
\]则
\[
\sqrt{s^TQ_i^Rs}
=
\sqrt{r_i^2s^Ts}
=
r_i.
\]并且
\[
x_R(s)
=
p_i+\frac{r_i^2s}{r_i}
=
p_i+r_is.
\]这才是球的支持点。
5. 障碍物支持点是 E-G4
障碍物要取朝机器人方向的点，也就是沿 \(s\) 方向的最小投影。
核心椭球给出：
\[
o_j-\frac{Q_j^Os}{\sqrt{s^TQ_j^Os}}.
\]不确定性椭球给出：
\[
-\frac{U_js}{\sqrt{s^TU_js}}.
\]偏置球半径 \(\delta_j\) 给出：
\[
-\delta_js.
\]所以
\[
x_O(s)
=
o_j
-
\frac{Q_j^Os}{\sqrt{s^TQ_j^Os}}
-
\frac{U_js}{\sqrt{s^TU_js}}
-
\delta_js.
\]如果某个形状矩阵为零，比如 \(U_j=0\)，那一项直接删掉，不算 \(0/0\)。
6. 代入得到 E-G1 和 E-G5
计算投影差：
\[
s^T(x_O(s)-x_R(s)).
\]代入上面两个支持点：
\[
s^T(x_O-x_R)
=
s^T(o_j-p_i)
-
\sqrt{s^TQ_i^Rs}
-
\sqrt{s^TQ_j^Os}
-
\sqrt{s^TU_js}
-
\delta_j.
\]因此
\[
\boxed{
s^T(x_O(s)-x_R(s))=\phi_{ij}(s)
}
\]这就是 E-G5。
所以 \(\phi_{ij}(s)\) 的意义非常明确：
\[
\boxed{
\phi_{ij}(s)
=
\text{沿方向 }s\text{ 的机器人-障碍物投影间隙}
}
\]任意 \(s\) 只能给一个方向上的投影间隙；只有最优方向 \(s^\ast\) 才给真实最短距离。
7. 为什么要最大化 \(\phi_{ij}(s)\)
两个凸体之间的真实欧氏距离是
\[
\operatorname{dist}(\mathcal R_i,\mathcal O_j)
=
\min_{x_R\in\mathcal R_i,\ x_O\in\mathcal O_j}
\|x_O-x_R\|.
\]对任意单位方向 \(s\)，都有
\[
\phi_{ij}(s)
\le
s^T(x_O-x_R)
\le
\|x_O-x_R\|.
\]所以
\[
\phi_{ij}(s)
\le
\operatorname{dist}(\mathcal R_i,\mathcal O_j).
\]也就是说，每个方向的投影间隙都是一个下界。
当两个凸体不相交时，最近点对 \(x_R^\ast,x_O^\ast\) 的连线方向
\[
s^\ast=
\frac{x_O^\ast-x_R^\ast}
{\|x_O^\ast-x_R^\ast\|}
\]正好就是最优分离平面的法向。此时
\[
\phi_{ij}(s^\ast)
=
\|x_O^\ast-x_R^\ast\|.
\]因此
\[
\boxed{
c_{ij}
=
\max_{\|s\|=1}\phi_{ij}(s)
=
\operatorname{dist}(\mathcal R_i,\mathcal O_j)
}
\]在不相交时成立。
</details>

---

在给定单位方向 \(s\) 时，机器人证书朝向障碍物的支持点和障碍物朝向机器人的支持点分别为

\[
x_R(s)=p_i+\frac{Q_i^Rs}{\sqrt{s^TQ_i^Rs}},
\tag{E-G3}
\]

\[
x_O(s)=o_j
-\frac{Q_j^Os}{\sqrt{s^TQ_j^Os}}
-\frac{U_js}{\sqrt{s^TU_js}}
-\delta_js.
\tag{E-G4}
\]

式（E-G4）约定省略形状矩阵为零的支持分量；例如 \(U_j=0\) 时，不计算形式上为 \(0/0\) 的第三项，而是直接把该项记为零。

二者沿 \(s\) 的投影间隙满足

\[
s^T\bigl(x_O(s)-x_R(s)\bigr)=\phi_{ij}(s).
\tag{E-G5}
\]

因此最优 \(s^*\) 同时给出最近分离方向、障碍物支持点和切平面。机器人采用球证书时取 \(Q_i^R=r_i^2I\)，在单位球面上有 \(\sqrt{s^TQ_i^Rs}=r_i\)。若同时令 \(U_j=0\)，式（E-G1）退化为式（E-F2），一般支持和与一维最近点 KKT 得到同一个几何答案。

令 \(d=o_j-p_i\)，并把 \(Q_i^R,Q_j^O,U_j\) 统一记为 \(A_m\)。目标函数的欧氏梯度为

\[
g_E(s)=d-\sum_m\frac{A_ms}{\sqrt{s^TA_ms}}.
\tag{E-G6}
\]

由于 \(s\) 被限制在单位球面 \(S^2\)，真正的一阶残差是切空间梯度

\[
g_R(s)=\left(I-ss^T\right)g_E(s).
\tag{E-G7}
\]

因为 \(s\) 被限制在单位球面上：

\[
\|s\|_2=1.
\]

所以 \(s\) 不能像普通三维变量那样随便往任意方向走。它只能在球面上滑动，不能往球心外面或里面走。

<details>
<summary>梯度 E-G6 怎么来的</summary>
令
\[
d=o_j-p_i,
\]并把三个形状矩阵统一写成
\[
A_m\in\{Q_i^R,\ Q_j^O,\ U_j\}.
\]定义
\[
\rho_m(s)=\sqrt{s^TA_ms}.
\]那么
\[
\phi(s)
=
s^Td-\sum_m \rho_m(s)-\delta_j.
\]对
\[
\rho_m(s)=\sqrt{s^TA_ms}
\]求导：
\[
\nabla_s\rho_m(s)
=
\frac{A_ms}{\sqrt{s^TA_ms}}
=
\frac{A_ms}{\rho_m}.
\]所以
\[
\boxed{
g_E(s)
=
\nabla_s\phi(s)
=
d-\sum_m\frac{A_ms}{\rho_m}
}
\]这就是 E-G6。
9. Hessian E-G8/E-G9 怎么来
继续对
\[
\frac{A_ms}{\rho_m}
\]求导：
\[
\nabla_s\left(\frac{A_ms}{\rho_m}\right)
=
\frac{A_m}{\rho_m}
-
\frac{(A_ms)(A_ms)^T}{\rho_m^3}.
\]因为 \(\phi\) 里是减号，所以 Hessian 贡献为
\[
\boxed{
-
\frac{A_m}{\rho_m}
+
\frac{(A_ms)(A_ms)^T}{\rho_m^3}
}
\]于是
\[
\boxed{
H_E(s)
=
\sum_m
\left(
-
\frac{A_m}{\rho_m}
+
\frac{(A_ms)(A_ms)^T}{\rho_m^3}
\right)
}
\]这就是 E-G8/E-G9。
10. 为什么要 Riemannian 梯度
因为 \(s\) 不是任意三维向量，而被限制在单位球面：
\[
\|s\|=1.
\]单位球面上的可行微小变化 \(\eta\) 必须满足
\[
s^T\eta=0.
\]也就是 \(\eta\) 只能在切平面里动：
\[
T_sS^2
=
\{\eta\mid s^T\eta=0\}.
\]所以普通欧氏梯度 \(g_E\) 里，沿着 \(s\) 的径向分量不能用。要投影到切空间：
\[
P_s=I-ss^T.
\]因此
\[
\boxed{
g_R(s)=P_sg_E(s)
}
\]最优时切向梯度为零：
\[
\boxed{
g_R(s^\ast)=0
}
\]意思是：在单位球面上，已经没有任何切向方向可以继续增大 \(\phi\)。
</details>


<details>
<summary>为什么要投影到切空间</summary>
假设当前点是单位向量 \(s\)，我们给它一个很小扰动 \(\eta\)：

\[
s^+ = s+\eta.
\]

为了仍然留在单位球面上，需要近似满足：

\[
\|s+\eta\|^2=1.
\]

展开：

\[
(s+\eta)^T(s+\eta)=1.
\]

\[
s^Ts+2s^T\eta+\eta^T\eta=1.
\]

因为

\[
s^Ts=1,
\]

所以

\[
1+2s^T\eta+\eta^T\eta=1.
\]

忽略二阶小量 \(\eta^T\eta\)，得到一阶可行条件：

\[
\boxed{s^T\eta=0.}
\]

这说明：允许的移动方向 \(\eta\) 必须和 \(s\) 垂直。

也就是：

\[
\boxed{
\eta \in T_sS^2=\{\eta\mid s^T\eta=0\}
}
\]

这里 \(T_sS^2\) 就是单位球面在 \(s\) 点的切平面。


直观图像：

\[
\text{径向方向： } s
\]

\[
\text{可走方向： }\eta\perp s
\]

也就是：

\[
\begin{array}{c}
\text{不能这样走： } s+\alpha s
\\[4pt]
\text{只能这样走： } s+\alpha\eta,\quad s^T\eta=0
\end{array}
\]

因为 \(s+\alpha s\) 会改变长度：

\[
\|s+\alpha s\|
=
\|(1+\alpha)s\|
=
|1+\alpha|.
\]

它不再满足 \(\|s\|=1\)。


普通欧氏梯度 \(g_E\) 可以分解成两部分：

\[
g_E
=
\underbrace{(s^Tg_E)s}_{\text{径向分量}}
+
\underbrace{\left(g_E-(s^Tg_E)s\right)}_{\text{切向分量}}.
\]

径向分量：

\[
(s^Tg_E)s
\]

是沿着 \(s\) 的方向，会把点推出球面，所以不能作为球面上的有效优化方向。

真正能让 \(s\) 在球面上变化的是切向分量：

\[
g_R
=
g_E-(s^Tg_E)s.
\]

写成矩阵形式：

\[
g_R
=
(I-ss^T)g_E.
\]

所以定义

\[
\boxed{
P_s=I-ss^T
}
\]

\[
\boxed{
g_R=P_sg_E
}
\]

就是把欧氏梯度投影到切平面。
黎曼梯度（Riemannian Gradient）是普通欧氏空间中梯度概念在黎曼流形（Riemannian Manifold，即带有局部测距和角度结构的弯曲空间）上的推广。它指出了在弯曲表面上，函数值增长最快的切向方向。在普通的欧氏空间中，计算梯度直接指向空间增长最快的方向。但在像球面、正交矩阵空间等黎曼流形上，如果你随意沿着普通梯度方向迈出一步，新位置可能就跑出了流形表面的约束。因此，我们需要将普通的欧式梯度“修正”到流形表面上。


为什么 \(I-ss^T\) 是投影矩阵？

因为 \(s\) 是单位向量，所以任意向量 \(v\) 在 \(s\) 方向上的投影是：

\[
\operatorname{proj}_s(v)
=
(s^Tv)s
=
ss^Tv.
\]

因此去掉径向分量后：

\[
v_{\text{tangent}}
=
v-ss^Tv
=
(I-ss^T)v.
\]

并且它一定和 \(s\) 垂直：

\[
s^T(I-ss^T)v
=
s^Tv-s^Tss^Tv
=
s^Tv-1\cdot s^Tv
=
0.
\]

所以

\[
\boxed{
(I-ss^T)v
}
\]

一定在切空间里。

最优条件也因此不是

\[
g_E(s^\ast)=0,
\]

而是

\[
\boxed{
g_R(s^\ast)=0.
}
\]

也就是：

\[
\boxed{
(I-s^\ast {s^\ast}^T)g_E(s^\ast)=0.
}
\]

这表示：虽然欧氏梯度可能不为零，但它已经没有切向分量了；在球面上已经没有能继续增大目标函数的方向。
</details>

---

最优法向满足 \(g_R(s^*)=0\)。每个支持半径项对欧氏 Hessian 的贡献为

\[
-\frac{A_m}{\rho_m}
+\frac{(A_ms)(A_ms)^T}{\rho_m^3},
\qquad
\rho_m=\sqrt{s^TA_ms},
\tag{E-G8}
\]

从而

\[
H_E(s)=\sum_m
\left(
-\frac{A_m}{\rho_m}
+\frac{(A_ms)(A_ms)^T}{\rho_m^3}
\right).
\tag{E-G9}
\]
<details>

<summary>Hessian E-G8/E-G9 怎么来</summary>
继续对
\[
\frac{A_ms}{\rho_m}
\]求导：
\[
\nabla_s\left(\frac{A_ms}{\rho_m}\right)
=
\frac{A_m}{\rho_m}
-
\frac{(A_ms)(A_ms)^T}{\rho_m^3}.
\]因为 \(\phi\) 里是减号，所以 Hessian 贡献为
\[
\boxed{
-
\frac{A_m}{\rho_m}
+
\frac{(A_ms)(A_ms)^T}{\rho_m^3}
}
\]于是
\[
\boxed{
H_E(s)
=
\sum_m
\left(
-
\frac{A_m}{\rho_m}
+
\frac{(A_ms)(A_ms)^T}{\rho_m^3}
\right)
}
\]这就是 E-G8/E-G9。
</details>

---

在切空间中，Riemannian Hessian 可写为

\[
H_R(s)
=P_s\left(H_E(s)-\bigl(s^Tg_E(s)\bigr)I\right)P_s,
\qquad P_s=I-ss^T.
\tag{E-G10}
\]

<details>
<summary>公式推导</summary>
目标函数记作
\[
f(s)=\phi_{ij}(s),
\qquad \|s\|=1.
\]普通欧氏梯度是
\[
g_E(s)=\nabla f(s).
\]但是 \(s\) 被限制在球面上，所以真正可用的梯度是投影后的：
\[
g_R(s)=P_sg_E(s),
\qquad
P_s=I-ss^T.
\]现在 Hessian 就是问：
\[
\boxed{
s \text{ 沿切向 } \eta \text{ 动一点时，} g_R(s) \text{ 怎么变？}
}
\]从
\[
g_R(s)=P_sg_E(s)
\]开始求导。令 \(\eta\) 是切向方向：
\[
s^T\eta=0.
\]对 \(g_R\) 求方向导数：
\[
Dg_R(s)[\eta]
=
D(P_sg_E(s))[\eta].
\]乘积求导：
\[
Dg_R(s)[\eta]
=
DP_s[\eta]g_E(s)
+
P_sDg_E(s)[\eta].
\]其中
\[
Dg_E(s)[\eta]=H_E(s)\eta.
\]所以第二项是：
\[
P_sH_E(s)\eta.
\]关键是第一项。因为
\[
P_s=I-ss^T,
\]所以
\[
DP_s[\eta]
=
-(\eta s^T+s\eta^T).
\]于是
\[
DP_s[\eta]g_E
=
-\eta(s^Tg_E)-s(\eta^Tg_E).
\]再投影回切空间：
\[
P_sDP_s[\eta]g_E
=
-(s^Tg_E)\eta.
\]因为
\[
P_s\eta=\eta,
\qquad
P_ss=0.
\]因此球面上的 Hessian 作用在 \(\eta\) 上就是：
\[
H_R(s)\eta
=
P_sH_E(s)\eta
-
(s^Tg_E(s))\eta.
\]写成矩阵形式：
\[
\boxed{
H_R(s)
=
P_s
\left(
H_E(s)-(s^Tg_E(s))I
\right)
P_s
}
\]这就是那个公式。
直观上它分三部分：
\[
\boxed{
H_E(s)
}
\]是普通三维空间里的 Hessian；
\[
\boxed{
-(s^Tg_E(s))I
}
\]是球面约束带来的曲率修正；
\[
\boxed{
P_s(\cdot)P_s
}
\]表示输入方向和输出方向都只取切空间部分。
所以它不是“直接列公式”，而是从
\[
g_R(s)=P_sg_E(s)
\]对 \(s\) 求导推出来的。
</details>

---

实现选择两个正交切向基 \(T=[t_1,t_2]\)，构造 \(2\times2\) 矩阵 \(T^TH_RT\)，并求 Newton 方向

\[
\eta=-T\left(T^TH_RT\right)^{-1}T^Tg_R.
\tag{E-G11}
\]

<details>
<summary>公式推导</summary>
这里的“原来的三维 Newton 方程”指的是在 \(\mathbb R^3\) 里写出来的 Newton 方程：
\[
H_R(s)\eta=-g_R(s).
\]其中：
\[
s\in\mathbb R^3,\qquad
g_R(s)\in\mathbb R^3,\qquad
H_R(s)\in\mathbb R^{3\times 3},
\qquad
\eta\in\mathbb R^3.
\]所以表面上它是三维方程。
但问题是：\(\eta\) 不能随便是任意三维方向，因为 \(s\) 必须留在单位球面上：
\[
\|s\|=1.
\]因此合法的微小更新方向 \(\eta\) 必须在切平面里：
\[
s^T\eta=0.
\]也就是：
\[
\eta\in T_sS^2.
\]球面 \(S^2\) 是二维曲面，所以它的切空间只有两个自由度。
于是我们选两个切向单位基：
\[
T=
\begin{bmatrix}
|&|\\
t_1&t_2\\
|&|
\end{bmatrix}
\in\mathbb R^{3\times 2},
\]满足：
\[
t_1^Ts=0,\qquad t_2^Ts=0,
\]\[
T^TT=I_2.
\]任何合法的切向更新都可以写成：
\[
\eta=u_1t_1+u_2t_2.
\]矩阵形式就是：
\[
\eta=Tu,
\qquad
u=
\begin{bmatrix}
u_1\\u_2
\end{bmatrix}.
\]这里的意思是：
\[
\boxed{
\text{不用直接求三维向量 }\eta，
\text{只求它在两个切向基上的坐标 }u
}
\]把
\[
\eta=Tu
\]代入三维 Newton 方程：
\[
H_R\eta=-g_R,
\]得到：
\[
H_RTu=-g_R.
\]这个式子左边还是三维向量。为了只看它在切平面两个方向上的分量，左乘 \(T^T\)：
\[
T^TH_RTu=-T^Tg_R.
\]这就是那个二维方程：
\[
\boxed{
T^TH_RT u=-T^Tg_R
}
\]然后解：
\[
\boxed{
u=-(T^TH_RT)^{-1}T^Tg_R
}
\]最后再转回三维更新方向：
\[
\boxed{
\eta=Tu
}
\]
</details>

---

只有切空间 Hessian 对最大化问题为负定、矩阵非奇异且 \(g_R^T\eta>0\) 时才接受 Newton 方向；否则回退到归一化的 Riemannian 梯度方向。候选法向通过

\[
s^+=\frac{s+\alpha\eta}{\|s+\alpha\eta\|_2}
\tag{E-G12}
\]

重新投影到单位球面，并用 Armijo 回溯选择 \(\alpha\)。上一周期同一 robot—proxy pair 的法向作为热启动；若最终 \(\|g_R\|\) 超过冻结的 KKT 残差门，则该法向不得进入 QP。这个受保护 Riemannian Newton 解决的是一般支持和问题，不与式（E-B6）的一维 Newton＋二分对同一个纯球—椭球 pair 重复执行。

<details>
<summary>公式推导</summary>
这句话是在说：**Newton 方向有时不靠谱，所以接受它之前要验货。**

我们现在要求的是

\[
\max_{\|s\|=1}\phi(s).
\]

在当前点 \(s\) 的切空间里，Newton 步长 \(\eta\) 是通过局部二次模型算出来的：

\[
\eta
=
-T(T^TH_RT)^{-1}T^Tg_R.
\]

但这个方向只有在局部模型看起来像“山顶”时才适合最大化。

---

**1. 为什么 Hessian 要负定**

最大化问题附近，目标函数的二次近似应该是：

\[
\phi(s+\eta)
\approx
\phi(s)+g_R^T\eta+\frac12\eta^TH_R\eta.
\]

如果是在局部最大点附近，曲率应该向下弯：

\[
\eta^TH_R\eta<0
\]

对所有非零切向 \(\eta\) 都成立。

这叫负定：

\[
\boxed{
H_R \prec 0
}
\]

在二维切空间里实际检查的是：

\[
\boxed{
T^TH_RT \prec 0
}
\]

意思是：从切平面任何方向走，二次模型都是“向下开的”。

如果 Hessian 不是负定，它可能像鞍点或者谷底：

\[
\eta^TH_R\eta>0
\]

这种情况下 Newton 方向可能把你带向错误位置。

---

**2. 为什么矩阵要非奇异**

Newton 方向需要求逆：

\[
(T^TH_RT)^{-1}.
\]

如果

\[
T^TH_RT
\]

是奇异矩阵，逆不存在。

也就是无法稳定地算：

\[
u=-(T^TH_RT)^{-1}T^Tg_R.
\]

所以必须要求：

\[
\boxed{
\det(T^TH_RT)\neq 0
}
\]

---

**3. 为什么还要检查 \(g_R^T\eta>0\)**

因为我们是最大化。

沿着方向 \(\eta\) 走一小步，目标函数一阶变化是：

\[
\phi(s+\alpha\eta)
\approx
\phi(s)+\alpha g_R^T\eta.
\]

如果

\[
g_R^T\eta>0,
\]

说明小步走过去目标值会上升：

\[
\boxed{
\phi \text{ 会变大}
}
\]

这叫上升方向。

如果

\[
g_R^T\eta\le 0,
\]

说明它不是上升方向，甚至可能让 \(\phi\) 变小，那就不能接受。

---

**4. 为什么回退到归一化 Riemannian 梯度方向**

如果 Newton 方向不满足条件，就用最稳的上升方向：

\[
\eta
=
\frac{g_R}{\|g_R\|}.
\]

因为梯度方向天然满足：

\[
g_R^T\eta
=
g_R^T\frac{g_R}{\|g_R\|}
=
\|g_R\|>0.
\]

只要

\[
g_R\neq 0,
\]

它一定是上升方向。

所以这句话的意思是：

\[
\boxed{
\text{能安全用 Newton 就用 Newton；Newton 不可靠时，就用梯度上升方向。}
}
\]

---

把原文翻译成算法语言就是：

```text
计算 Newton 方向 eta

如果：
1. T^T H_R T 是负定的
2. T^T H_R T 可逆
3. g_R^T eta > 0，也就是 eta 是上升方向

那么：
    接受 Newton 方向

否则：
    eta = g_R / ||g_R||
```

这不是重新推导几何，而是在保证数值优化别走歪。
</details>


<details>
<summary>逐句翻译</summary>
这段其实全是**实现层面的保护措施**，不是新的几何推导。逐句翻译一下。

**1. “重新投影到单位球面”**

因为 \(s\) 必须满足：

\[
\|s\|=1.
\]

但是更新时先算：

\[
s+\alpha\eta.
\]

这个向量一般不再刚好是单位长度，所以要归一化：

\[
s^+
=
\frac{s+\alpha\eta}{\|s+\alpha\eta\|}.
\]

意思是：

\[
\boxed{
\text{走一步之后，把 }s\text{ 拉回单位球面上}
}
\]

---

**2. “用 Armijo 回溯选择 \(\alpha\)”**

\(\alpha\) 是步长。

如果步子太大，可能目标函数反而下降，所以先试：

\[
\alpha=1.
\]

如果

\[
\phi(s^+)>\phi(s)
\]

提升不够，就缩小：

\[
\alpha\leftarrow \beta\alpha,
\qquad 0<\beta<1.
\]

比如：

\[
1,\ 0.5,\ 0.25,\ 0.125,\ldots
\]

直到目标函数确实有足够上升。

所以：

\[
\boxed{
\text{Armijo 回溯就是：步子太大就不断减半，直到这一步靠谱}
}
\]

---

**3. “上一周期同一 robot—proxy pair 的法向作为热启动”**

robot—proxy pair 就是某个机器人证书和某个障碍证书这一对：

\[
(\text{robot }i,\ \text{obstacle }j).
\]

每个控制周期都要重新算它们之间的分离方向。

如果上一周期已经算出了：

\[
s^\ast_{\text{old}},
\]

那下一周期不用从随机方向开始，而是直接用上一次的结果当初值：

\[
s_0=s^\ast_{\text{old}}.
\]

这叫热启动。

原因是机器人和障碍物每个周期变化通常很小，所以最优法向也变化不大。

\[
\boxed{
\text{热启动 = 用上一次的答案当这一次的初始猜测}
}
\]

---

**4. “若最终 \(\|g_R\|\) 超过冻结的 KKT 残差门”**

最优条件是：

\[
g_R(s^\ast)=0.
\]

实际数值计算不可能完全等于零，所以要求：

\[
\|g_R\| \le \varepsilon_{\text{KKT}}.
\]

这里 \(\|g_R\|\) 就是“离最优还差多少”的指标。

如果最后算完发现：

\[
\|g_R\|>\varepsilon_{\text{KKT}},
\]

说明这个 \(s\) 没有收敛，分离方向不可信。

所谓“冻结的 KKT 残差门”，就是一个固定阈值，例如：

\[
\varepsilon_{\text{KKT}}=10^{-5}.
\]

---

**5. “该法向不得进入 QP”**

LiuQP 最后会把障碍约束放进 QP，例如：

\[
h_{ij}(q)\ge 0.
\]

这个约束需要用法向 \(s^\ast\) 来构造。

如果 \(s^\ast\) 没有收敛，就不能把它送进 QP，否则 QP 会基于错误的方向做避障。

所以：

\[
\boxed{
\|g_R\|\text{ 太大 } \Rightarrow s^\ast\text{ 不可信 } \Rightarrow 不加入 QP}
}
\]

---

**6. “受保护 Riemannian Newton”**

意思是这个 Newton 法不是裸跑，而是带保护：

\[
\boxed{
\text{负定检查 + 上升方向检查 + Armijo 步长 + KKT 残差检查}
}
\]

所以叫“受保护”。

---

**7. “解决的是一般支持和问题”**

它解决的是：

\[
\max_{\|s\|=1}
\left[
s^T(o_j-p_i)
-
\sqrt{s^TQ_i^Rs}
-
\sqrt{s^TQ_j^Os}
-
\sqrt{s^TU_js}
-
\delta_j
\right].
\]

也就是一般情况：

\[
\boxed{
\text{机器人椭球 + 障碍椭球 + 不确定性椭球 + 偏置}
}
\]

---

**8. “不与 E-B6 的一维 Newton + 二分对同一个纯球—椭球 pair 重复执行”**

如果是简单情况：

\[
Q_i^R=r_i^2I,\qquad U_j=0,
\]

也就是：

\[
\boxed{
\text{机器人球} - \text{障碍椭球}
}
\]

前面 2.6 的方法已经可以算：

\[
F(\lambda)=0.
\]

那就用 2.6 的一维 Newton + 二分。

不要同时又用 2.7 的球面 Newton 再算一遍，因为二者求的是同一个距离，只是方法不同。

---

整段压缩成一句话就是：

\[
\boxed{
\text{2.7 的球面 Newton 是给一般椭球/不确定性情况用的，并且只有收敛可靠时才把结果放进 QP；纯球-椭球情况已经由 2.6 解决，不要重复计算。}
}
\]
</details>

---

## 2.8 IRIS-inspired 在本推导中的准确含义

本研究所称 IRIS-inspired，不是运行 IRIS 的构型空间采样、碰撞构型切除、最大体积内接椭球膨胀或区域连接图。这里借用的是一个更基础的凸几何思想：对当前机器人证书与障碍物凸证书求最近分离方向，以障碍物在该方向上的支持点构造法平面，并把机器人限制在平面的另一侧。式（E-D）和式（E-G3）—（E-G5）就是这一思想在 LiuQP 速度约束中的具体形式。

球形 LiuQP 的法向由中心连线闭式给出，障碍球支持点只需减去半径。椭球没有方向无关的统一半径，必须通过式（E-B6）求真实最近点，或通过式（E-G2）求支持和的最优法向。得到法向以后，进入 QP 的仍然只是一条线性半空间约束。IRIS-inspired 改变的是平面系数的几何来源，不把 QP 改成非线性规划，也不向控制器提供区域链或全局路径。

## 2.9 用椭球支持最小值推广母版的 erase-remove

母版在为机器人模块 \(i\) 和障碍球 \(j\) 构造切平面后，检查该平面是否也把另一个障碍球 \(k\) 完全挡在障碍侧。椭球版沿用同一逻辑。令 \(s=\tilde s_{ij}^{*}\)，令 \(x_{O,j}=x_O(s)\) 为式（E-G4）给出的第 \(j\) 个障碍代理支持点，则分离平面为

\[
\mathcal P_{ij}^{E}
=\{x:s^T(x-x_{O,j})=0\}.
\tag{E-H1}
\]

机器人球完全位于平面负侧的条件是

\[
\max_{x\in\mathcal B_i}
s^T(x-x_{O,j})
=s^T(p_i-x_{O,j})+r_i\le0.
\tag{E-H2}
\]

若把控制安全余量也作为当前平面的保护带，则判据相应写为

\[
s^T(p_i-x_{O,j})+r_i+d_{\mathrm{safe}}\le0.
\tag{E-H3}
\]

对另一个障碍代理
\(\mathcal K_k=E(o_k,Q_k)\oplus E(0,U_k)\oplus B(0,\delta_k)\)，其在方向 \(s\) 上相对平面的最小投影为

\[
\begin{aligned}
m_k(s)
&=\min_{x\in\mathcal K_k}s^T(x-x_{O,j})\\
&=s^T(o_k-x_{O,j})
-\sqrt{s^TQ_ks}
-\sqrt{s^TU_ks}
-\delta_k.
\end{aligned}
\tag{E-H4}
\]

若

\[
m_k(s)\ge0,
\tag{E-H5}
\]

则第 \(k\) 个椭球完全位于平面正侧。任何从机器人侧到达 \(\mathcal K_k\) 的连续运动都必须先穿过 \(\mathcal P_{ij}^{E}\)，所以约束 \(j\) 已经对 \(k\) 提供遮挡，\(k\) 对当前机器人球和当前控制周期是冗余的。若 \(m_k(s)<0\)，椭球与平面相交、跨过平面或部分落到机器人侧，便不能由该平面删除。

对每个机器人球 \(i\)，从当前候选环境代理集合 \(\mathcal S^E\) 出发，按冻结的距离下界和稳定 ID 顺序构造平面并执行有序 erase-remove，得到

\[
\mathcal S_i^E\subseteq\mathcal S^E.
\tag{E-H6}
\]

\(\mathcal S_i^E\) 是机器人球相关且周期相关的活动集合，不是代理生成阶段的“最少覆盖集合”。前者删除对当前碰撞约束被其他平面支配的代理，后者删除对环境表面覆盖没有唯一见证的代理，两种删除发生在不同数据层。CONTACT/RECOVERY pair 还必须保留自己的恢复行；即使另一平面在无碰状态下遮挡了它，也不能利用冗余判定隐藏已经接触或重叠的 pair。

## 2.10 椭球型综合 QP 与 LiuQP 三状态

令 \(a_{ij}=\tilde s_{ij}^TJ_{p,i}(q)\)。将硬任务等式、关节界、工作空间边界和椭球碰撞行组合，母版公式（24）的椭球对应式为

\[
\begin{aligned}
\min_{\dot q}\quad
&\frac12\dot q^T\dot q,\\
\mathrm{s.t.}\quad
&J\dot q=b,\\
&\frac{q_{\min}-q}{\Delta t}
\le\dot q\le
\frac{q_{\max}-q}{\Delta t},\\
&\dot q_{\min}\le\dot q\le\dot q_{\max},\\
&\hat s_{if}^{T}J_{p,i}\dot q
\le\frac{d_{if}-r_i}{\Delta t},\\
&a_{ik}\dot q
\le\frac{h_{ik}}{\Delta t},
\quad
\forall (i,k)\in\mathcal V_K\times\mathcal S_i^E.
\end{aligned}
\tag{24E}
\]

在一个控制周期内，\(J\)、\(J_{p,i}\)、\(\tilde s_{ik}\)、\(h_{ik}\) 和 \(\mathcal S_i^E\) 都是由当前状态预先计算的常量，所以式（24E）仍是关于 \(\dot q\) 的凸 QP。椭球最近点计算变复杂不会破坏 QP 的凸性；它增加的是组装 QP 之前的窄阶段计算量。

当硬任务等式与安全或关节界冲突时，母版公式（25）把任务追踪移到目标函数：

\[
f(\dot q)
=\|\dot q\|_2^2
+\lambda\|J\dot q-b\|_2^2.
\tag{25}
\]

展开并忽略常数项后，标准形式的 Hessian 与梯度为

\[
H_0=2\left(I+\lambda J^TJ\right),
\qquad
g_0=-2\lambda J^Tb.
\tag{E-I1}
\]

任务项因此是软的，式（23E）的碰撞行仍然是硬的。三状态必须依据尚未减去安全余量的原始证书间隙 \(c_{ij}\) 分类，而不能依据 \(h_{ij}\) 分类。令近障碍阈值为 \(d_{\min}>0\)，则 NORMAL、NEAR 与 CONTACT/RECOVERY 分别满足

\[
\begin{cases}
c_{ij}\ge d_{\min}, & \mathrm{NORMAL},\\
0<c_{ij}<d_{\min}, & \mathrm{NEAR},\\
c_{ij}\le0, & \mathrm{CONTACT/RECOVERY}.
\end{cases}
\tag{E-I2}
\]

NORMAL 状态只使用基础硬碰撞行。NEAR 状态保留同一硬行，并将母版公式（26）中的球面法向替换为椭球真实最近法向：

\[
f(\dot q)
=\|\dot q\|_2^2
+\lambda\|J\dot q-b\|_2^2
+\mu_{ij}\left\|a_{ij}\dot q\right\|_2^2.
\tag{26E}
\]

附加项对 Hessian 的贡献为

\[
H_{\mathrm{near},ij}=2\mu_{ij}a_{ij}^Ta_{ij}\succeq0.
\tag{E-I3}
\]

因为法向速度被平方，式（26E）同时惩罚朝向障碍物和离开障碍物的法向运动，使机器人更偏向沿切平面运动。该项是软代价，不是碰撞松弛变量，也不会把硬安全行变成可违反约束。

进入 CONTACT/RECOVERY 后，NEAR 软惩罚被移除，母版公式（27）推广为

\[
a_{ij}\dot q\le\gamma_{ij},
\qquad \gamma_{ij}<0.
\tag{27E}
\]

由于 \(\tilde s_{ij}\) 从机器人指向障碍物，\(a_{ij}\dot q<0\) 表示机器人沿相反方向撤离。当前正式实现保留基础硬碰撞行 \(a_{ij}\dot q\le h_{ij}/\Delta t\)，并额外加入式（27E）的恢复行；两者中更严格的一条决定本周期允许速度。多个恢复要求若相互冲突，QP 可能报告 primal infeasible，此时规定的失败行为是输出精确零速度并记录失败，而不是调用随机扰动、路点或全局规划器。

把所有 NEAR pair 的贡献合并后，当前软任务 QP 可概括为

\[
\begin{aligned}
\min_{\dot q}\quad
&\frac12\dot q^T
\left(H_0+\sum_{(i,j)\in\mathcal N}
2\mu_{ij}a_{ij}^Ta_{ij}\right)\dot q
+g_0^T\dot q,\\
\mathrm{s.t.}\quad
&\text{关节位置界、速度界、工作空间界、式（23E）硬行，}\\
&a_{ij}\dot q\le\gamma_{ij},
\quad (i,j)\in\mathcal C.
\end{aligned}
\tag{E-I4}
\]

其中 \(\mathcal N\) 与 \(\mathcal C\) 分别是当前 NEAR 和 CONTACT/RECOVERY pair 集。椭球型 LiuQP 与球型 LiuQP 使用同一个 QP 结构；二者的差别只体现在 \(\tilde s_{ij}\)、\(c_{ij}\)、\(h_{ij}\) 和有序删除后留下的活动 pair。

## 2.11 周期级算法及输入到输出的数据流

第 \(k\) 个控制周期开始时，控制器读取当前关节状态 \(q_k\)、已完整发布的环境代理快照和唯一任务目标。运动学模块利用母版公式（1）—（13）计算末端与全部机器人证书球的中心、点雅可比；任务模块利用公式（14）—（17）得到 \(b_k\)；关节边界由公式（19）—（20）转换为本周期速度上下界。

碰撞模块对每个机器人球查询当前候选代理。对 \(U=0\) 的单椭球 pair，按式（E-B6）用热启动受保护 Newton 和必要二分得到 \(x^*\) 与 \(\tilde s\)；对一般支持和 pair，按式（E-G2）与式（E-G6）—（E-G12）求最优法向。通过残差门的结果产生切平面、\(c\)、\(h\) 和状态分类，再由式（E-H4）执行 ordered erase-remove。未通过残差门的法向不能进入 QP，也不能退化成中心连线近似继续运行。

活动平面被装配为式（E-I4），求解器返回 \(\dot q_k\) 后直接执行一步

\[
q_{k+1}=q_k+\Delta t\dot q_k.
\tag{E-J1}
\]

随后全部状态相关量重新计算。整个闭环可表示为

\[
q_k
\rightarrow(p_i,J_{p,i},b_k)
\rightarrow(x^*_{ij},\tilde s_{ij},c_{ij})
\rightarrow\mathcal S_{i,k}^E
\rightarrow\mathrm{QP}_k
\rightarrow\dot q_k
\rightarrow q_{k+1}.
\tag{E-J2}
\]

式（E-J2）没有完整参考路径。每个 QP 只在当前构型对碰撞几何做一阶限制，所以算法属于顺序凸的局部在线控制/规划方法，不具有一般非凸构型空间的全局完备性。椭球表示可以恢复被球证书各向同性膨胀占据的空间，但不能据此推导其会消除所有局部平衡或改变所有同伦障碍。

## 2.12 与母版球形推导的逐项对应

| 母版球形对象 | 椭球型对应对象 | 是否改变 QP 结构 |
|---|---|---|
| 公式（1）—（20）：运动学、点速度、任务反馈、关节界 | 原样复用 | 否 |
| 公式（21）—（22）：工作空间平面 | 原样复用，并显式除以 \(\Delta t\) | 否 |
| 补充式 A：球—球中心距离 | 式（E-A5）的球—椭球集合距离 | 只改变几何预计算 |
| 补充式 B、C：中心连线和球面最近点 | 式（E-B6）—（E-B11）的真实椭球最近点与法向 | 只改变窄阶段 |
| 补充式 D：球面切平面 | 式（E-D）或式（E-H1）的椭球支持平面 | 否 |
| 补充式 E、公式（23）：球面间隙和速度行 | 式（E-F1）—（23E） | 行结构不变，系数改变 |
| 补充式 F、G：冗余球平面判定 | 式（E-H2）—（E-H5）的椭球支持最小值 | 删除公式改变 |
| 补充式 H：\(\mathcal S_i\subseteq\mathcal S\) | 式（E-H6）的 \(\mathcal S_i^E\subseteq\mathcal S^E\) | 否 |
| 公式（24）：综合硬任务 QP | 公式（24E） | 否 |
| 公式（25）：软任务目标 | 原样复用 | 否 |
| 公式（26）：近障碍球面法向惩罚 | 公式（26E）的真实椭球法向惩罚 | Hessian 形式不变 |
| 公式（27）：球面接触恢复 | 公式（27E）的椭球接触恢复 | 行结构不变 |

这组对应关系说明，椭球型 LiuQP 并不是另一个控制器。它保留原 LiuQP 的速度级 QP、三状态和周期级重建，只把球几何提供的闭式法向与半径差替换成椭球支持几何提供的真实法向与间隙。

## 2.13 数学成立范围与实现审计点

一维乘子推导假定机器人球心位于障碍核心椭球外部。机器人球与椭球发生接触时，球心通常仍在核心外部，因此 \(c_{ij}\le0\) 并不意味着 \(p_i\in\mathcal E_j\)。若球心真正进入核心椭球，\(\lambda\ge0\) 的外点最近点分支不再给出严格的内部有符号距离；实现必须进入单独的穿透恢复处理或拒收该周期，不能把任意中心方向称为精确最近法向。当前代码对核心内部点存在回退分支，代码审核时应把它明确标为恢复保护，而不是式（E-B6）的精确外点解。

式（23E）采用一步 Euler 线性化，适用于静态障碍与足够短的控制周期。动态障碍需要在 \(\dot c\) 中加入障碍支持点速度；机器人若改用随连杆转动的椭球，还需要加入 \(\dot Q_i^R\) 对支持半径的贡献。当前主对照中环境静态、机器人保持球证书，因此这两项不进入 QP。

支持和分离公式证明的是当前凸证书之间的分离，不自动证明点云代理完整覆盖真实障碍。覆盖性必须由 CenterVox 代表点误差、候选构造和连续表面证书另行给出；MVT、AABB 和 SIMD 也只负责在不漏候选的条件下加速找到需要执行本章窄阶段的 pair。它们将在后续章节推导，不应混入本章距离公式。

本章得到的核心结论是：母版 LiuQP 中从模块点速度到 QP 的线性结构可以完整保留；球改为椭球后，新增计算集中在 QP 之前的真实最近点或最优支持法向求解。只要这些几何量通过残差门，式（23E）、（26E）、（27E）仍然关于 \(\dot q\) 线性或二次，当前周期仍是凸 QP。椭球带来的通道优势来源于 \(\sqrt{s^TQs}\) 的方向相关支持半径，而不是取消安全余量、遗漏代理或向控制器注入路径信息。
