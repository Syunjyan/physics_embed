# 简化模型约束说明

## 1. 定义

本文档采用更严格的定义：

```text
简化模型 / reduced model = 原 PDE 在特定场景假设下得到的近似方程或解析近似解。
```

因此，下列内容不再归入“经验公式”或“简化模型”：

- Hooke 定律：线弹性本构关系，是完整线弹性 PDE 系统的一部分；
- Fourier 定律：热传导本构关系，是热通量与温度梯度的闭合关系；
- Robin 换热边界：边界模型，不是原 PDE 的场方程简化；
- 能量衰减：物理正则或后处理检查，不是简化 PDE。

## 2. 当前实现

代码文件：

```text
physics_embed/reduced_models.py
```

包含：

- `heat_1d_steady_conduction`
- `burgers_diffusion_dominant`
- `burgers_linearized_advection_diffusion`
- `navier_stokes_stokes_limit`
- `navier_stokes_poiseuille_channel`
- `navier_stokes_bernoulli_inviscid`
- `linear_elasticity_uniaxial_stress`

这些函数都返回残差字典，可通过 `ReducedModelResidual` 或消融脚本加入训练。

## 3. 四类方程的简化模型

### Heat

一维稳态无源导热：

```text
d2T/dx2 = 0
T(x) = T_hot + (T_cold - T_hot) x/L
```

适用条件：无源项、常导热系数、一维传热、两端定温。

### Burgers

扩散主导近似：

```text
u_t ≈ nu Delta u
v_t ≈ nu Delta v
```

适用条件：速度幅值小，非线性对流项相对黏性扩散可忽略。

线性化对流-扩散：

```text
u_t + U0 u_x + V0 u_y ≈ nu Delta u
```

适用条件：围绕常值基流的小扰动。

### Navier-Stokes

Stokes 极限：

```text
grad(p) ≈ nu Delta u
```

适用条件：低 Reynolds 数，惯性项可忽略。

Poiseuille 通道流：

```text
u(y) = 4 Umax y(H-y)/H^2
v = 0
```

适用条件：稳态、层流、平行板通道、充分发展流。

Bernoulli 无粘近似：

```text
p + 0.5 rho |u|^2 = const
```

适用条件：稳态、无粘、不可压、沿同一流线。

### Linear Elasticity

单轴应力近似：

```text
sigma_x = E epsilon_x
epsilon_y = -nu epsilon_x
sigma_y = 0
tau_xy = 0
```

适用条件：细长试样、单轴拉压、远离夹具和应力集中区域。

本项目新增了专门匹配该假设的数据集：

```text
linear_elasticity_uniaxial
data/linear_elasticity_uniaxial.npz
```

注意：Hooke 定律本身不是简化模型，而是线弹性 PDE 的本构闭合关系。

## 4. 消融实验

简化模型消融脚本位于：

```text
experiments/heat_reduced_model_ablation/run_ablation.py
experiments/burgers_reduced_model_ablation/run_ablation.py
experiments/navier_stokes_reduced_model_ablation/run_ablation.py
experiments/linear_elasticity_reduced_model_ablation/run_ablation.py
```

其中线弹性 reduced-model ablation 默认使用 `linear_elasticity_uniaxial` 数据集，而不是原来的波动型 MMS 线弹性数据集。

线弹性 Hooke 本构消融脚本单独放在：

```text
experiments/linear_elasticity_constitutive_ablation/run_ablation.py
```

它回答的是“完整线弹性 PINN 是否需要本构残差”，不是经验公式消融。

