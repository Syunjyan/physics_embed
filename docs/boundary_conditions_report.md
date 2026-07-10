# PINN 边界条件增强约束报告

## 1. 问题背景

标准 PINN 通常把边界条件写成损失项：

```text
L = L_pde + lambda_bc * L_bc + lambda_data * L_data
```

这种方法简单，但存在两个问题：第一，边界 loss 和 PDE loss 的尺度可能不同，需要手工调权重；第二，边界条件只是在优化意义上满足，不保证每一步网络输出都满足边界。

如果我们进一步知道某种形式的边界解、局部值、通量或边界上的函数关系，可以把这些信息作为更强的约束嵌入 PINN。

## 2. 边界条件类型

常见边界条件包括：

- Dirichlet：直接给定解值，例如 `u=g(x)` 或位移 `u=0`；
- Neumann：给定法向导数或通量，例如 `grad(u).n=q`；
- Robin：给定解值与导数的线性组合，例如 `a*u+b*grad(u).n=c`；
- 周期边界：相对边界上的解值和导数相同；
- 局部观测：边界或内部少量点已知真实值。

## 3. 软约束方法

软约束是最通用方式。对边界点 `x_b` 加入：

```text
L_D = mean(|u_theta(x_b)-g(x_b)|^2)
L_N = mean(|grad(u_theta)(x_b).n-q(x_b)|^2)
L_R = mean(|a*u_theta+b*grad(u_theta).n-c|^2)
```

优点是适合任意边界和混合条件，容易扩展到热流、力、应力、速度入口、压力出口等。缺点是需要权重调节，且边界不一定精确满足。

本仓库当前四类方程都使用软边界约束：

- 热传导：左右 Dirichlet，上下 Neumann 热流；
- Burgers：初始条件和部分空间边界值；
- Navier-Stokes：初始条件和部分空间边界速度/压力；
- 线弹性：位移边界和应力/牵引边界。

## 4. 硬约束方法

对 Dirichlet 条件，如果边界函数 `g(x)` 已知，可以构造：

```text
u_theta(x) = G(x) + phi(x) * N_theta(x)
```

其中 `G(x)` 是满足边界条件的延拓函数，`phi(x)` 是在边界为 0、内部非零的距离函数或 bubble 函数。这样在边界上 `u_theta=g`，不需要再加 Dirichlet loss。

对于单位正方形上所有边界为零的情况，可用：

```text
phi(x,y)=x*(1-x)*y*(1-y)
u_theta(x,y)=phi(x,y)*N_theta(x,y)
```

若左右边界有非零温度，则先构造线性延拓 `G(x,y)`，再乘 bubble 学习内部修正。代码中的 `BoxDirichletTransform` 就是这个思路。

硬约束优点是边界精确满足、减少 loss 权重竞争；缺点是需要知道可微的边界延拓，并且对复杂几何、Neumann/Robin 条件更难。

## 5. Neumann 与 Robin 的增强方式

Neumann 和 Robin 条件可以继续使用软约束，也可以半硬约束：

1. 对输出做变换，使 Dirichlet 条件硬满足；
2. 对 Neumann/Robin 条件保留软残差；
3. 对边界附近增加采样密度，提升导数拟合；
4. 对边界法向导数使用更高权重或自适应权重。

对于热传导，对流换热可写成 Robin 残差：

```text
-k*grad(T).n - h*(T-T_inf) = 0
```

对于线弹性，牵引边界为：

```text
sigma*n = t_bar
```

对于 Navier-Stokes，入口速度、壁面 no-slip、出口压力都可以按边界类型分别构造残差。不可压条件可用流函数硬嵌入，速度边界仍可通过 soft 或 hard transform 处理。

## 6. 多阶段训练建议

没有物理背景时，可以采用保守的多阶段流程：

1. 阶段一：仅训练数据项和强边界项，让网络先学到边界和观测点的大致尺度。
2. 阶段二：加入 PDE 残差，逐步提高 PDE 权重，让内部场满足控制方程。
3. 阶段三：加入简化模型、对称性和全局守恒检查，做精修。
4. 阶段四：对边界残差、PDE 残差最大的区域重新采样，进行 residual-based adaptive refinement。

这种流程比一开始把所有 loss 混在一起更稳，尤其适合边界条件复杂、不同物理量尺度差异大的问题。

## 7. 针对四类方程的建议

### 热传导

如果已知边界温度，优先使用硬 Dirichlet 变换；如果已知热流或对流换热，用 Neumann/Robin 残差。后处理检查总热通量是否平衡。

### Burgers

初始条件可以使用时间硬约束，例如：

```text
u(x,y,t)=u0(x,y)+t*N_theta(x,y,t)
```

这样 `t=0` 自动满足初始场。空间边界可继续使用 soft loss 或周期约束。

### Navier-Stokes

不可压条件建议用流函数或投影结构硬嵌入。壁面 no-slip、入口速度和出口压力可以分开加权。若已知通道流近似剖面，可先用剖面预训练，再加入完整 NS 残差。

### 线弹性

固定位移边界适合硬约束；牵引边界适合软残差 `sigma*n=t`。若已知对称面，可在对称面施加法向位移为零或切向应力为零等条件。

## 8. 实施建议

当前仓库已提供三类基础组件：

- 方程自带 `boundary_loss`，用于普通软约束；
- `BoxDirichletTransform`，用于矩形域 Dirichlet 硬约束；
- `SymmetryResidual` 和 `ReducedModelResidual`，用于边界之外的先验增强。

下一步若要加强边界约束，可先从热传导开始，把左右温度边界从 soft loss 改成 hard transform；然后在线弹性中对固定位移边界使用 hard transform；最后再处理 Navier-Stokes 和 Burgers 的时间初始条件硬约束。
