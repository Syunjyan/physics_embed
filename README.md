# physics_embed

`physics_embed` 是一个面向二维物理方程的 PINN 实验仓库。第一阶段目标是为结构力学、热传导、Burgers 方程和 Navier-Stokes 方程生成可验证数据集，并提供统一 PINN 训练入口。第二阶段目标是在 PINN 前后加入对称性、简化模型和更强边界条件约束模块。

## 当前方程

- `heat`: 二维稳态热传导，形式为 `-Delta u = f`。
- `burgers`: 二维粘性 Burgers 方程，输出速度分量 `u, v`。
- `navier_stokes`: 二维不可压 Navier-Stokes 方程，网络输出流函数 `psi` 和压力 `p`，速度由自动微分得到。
- `linear_elasticity`: 二维线弹性静力学，输出位移和应力，残差包含平衡方程、几何方程和本构关系。

当前数据集采用 manufactured solution method (MMS, 制造解方法) 生成。这样每个方程都有解析真值、源项或体力项、边界条件，便于验证数据生成和 PINN 残差实现。

## 数据集物理场景与参数

所有数据集第一版都定义在二维单位区域 `[0, 1] x [0, 1]`。非稳态问题额外使用时间区间 `t in [0, 1]`。当前数据不来自真实 CFD/FEM 求解器，而是用制造解生成，因此更适合验证 PINN 框架、PDE 残差、边界约束和先验嵌入模块。

### `heat`: 二维稳态热传导

物理场景：单位方形薄板稳态导热，左右边界给定温度，上下边界给定热流。内部存在源项，使解析温度场满足 Poisson 型热传导方程。

控制方程：

```text
-Delta u = f
u(x,y) = sin(pi*x) sin(pi*y)
f(x,y) = 2*pi^2 sin(pi*x) sin(pi*y)
```

边界条件：

```text
x=0, x=1: u = 0
y=0: du/dy = pi*sin(pi*x)
y=1: du/dy = -pi*sin(pi*x)
```

场变量：温度 `u`，源项 `f`。

### `burgers`: 二维粘性 Burgers 对流-扩散

物理场景：二维速度场在非线性自对流和粘性扩散共同作用下演化，可作为 Navier-Stokes 的简化非线性测试问题。

控制方程：

```text
u_t + u*u_x + v*u_y - nu*Delta u = f_u
v_t + u*v_x + v*v_y - nu*Delta v = f_v
```

参数：

```text
nu = 0.01
```

制造解：

```text
u(x,y,t) = sin(pi*x) sin(pi*y) exp(-t)
v(x,y,t) = cos(pi*x) sin(pi*y) exp(-t)
```

场变量：速度分量 `u, v`，源项 `f_u, f_v`。

### `navier_stokes`: 二维不可压 Navier-Stokes

物理场景：二维不可压黏性流体的非定常速度-压力场。网络不直接输出速度，而是输出流函数 `psi` 和压力 `p`，速度由自动微分得到，从结构上满足不可压连续性。

控制方程：

```text
u_t + u*u_x + v*u_y + p_x - nu*Delta u = f_u
v_t + u*v_x + v*v_y + p_y - nu*Delta v = f_v
u_x + v_y = 0
```

参数：

```text
nu = 0.01
```

制造解：

```text
psi(x,y,t) = sin(pi*x) sin(pi*y) exp(-t)
p(x,y,t) = sin(pi*x) cos(pi*y) exp(-t)
u = d psi / dy
v = -d psi / dx
```

场变量：流函数 `psi`、压力 `p`、速度 `u, v`、动量源项 `f_u, f_v`。

### `linear_elasticity`: 二维线弹性静力学

物理场景：单位方形各向同性线弹性体的小变形静力问题，包含位移、应变、应力、体力和混合边界条件。

控制关系：

```text
div(sigma) + f = 0
epsilon = 0.5 * (grad(u) + grad(u)^T)
sigma = lambda * tr(epsilon) * I + 2 * mu * epsilon
```

材料与制造解参数：

```text
lambda = 1.0
mu = 0.5
Q = 4.0
ux = cos(2*pi*x) sin(pi*y)
uy = sin(pi*x) * Q*y^4/4
```

边界条件：

```text
y=0: ux=0, uy=0
x=0, x=1: uy=0, sigmaxx=0
y=1: ux=0, sigmayy=(lambda+2*mu)*Q*sin(pi*x)
```

场变量：位移 `ux, uy`，应力 `sigmaxx, sigmayy, sigmaxy`，应变 `exx, eyy, exy`，体力 `fx, fy`。

### `linear_elasticity_uniaxial`: 单轴拉伸线弹性

物理场景：单位矩形试样沿 x 方向单轴拉伸，横向 y 方向因泊松效应收缩。该数据集用于验证“单轴应力近似”这类简化模型约束。

近似假设：

```text
sigmaxx = E * exx
eyy = -nu * exx
sigmayy = 0
sigmaxy = 0
```

参数：

```text
E = 1.0
nu = 0.3
alpha = 0.02
```

解析场：

```text
ux = alpha * x
uy = -nu * alpha * y
sigmaxx = E * alpha
sigmayy = 0
sigmaxy = 0
```

场变量：位移 `ux, uy`，应力 `sigmaxx, sigmayy, sigmaxy`，应变 `exx, eyy, exy`，体力 `fx, fy`。

## 安装

```bash
pip install -e .
```

## 生成数据集

```bash
python scripts/generate_dataset.py --equation heat --out data/heat.npz
python scripts/generate_dataset.py --equation burgers --out data/burgers.npz
python scripts/generate_dataset.py --equation navier_stokes --out data/navier_stokes.npz
python scripts/generate_dataset.py --equation linear_elasticity --out data/linear_elasticity.npz
python scripts/generate_dataset.py --equation linear_elasticity_uniaxial --out data/linear_elasticity_uniaxial.npz
```

可调参数：

- `--spatial-resolution`: 单方向空间网格数，默认 `64`。
- `--time-steps`: 时间步数，仅对 Burgers 和 Navier-Stokes 有效，默认 `21`。

## 训练 PINN

```bash
python scripts/train_pinn.py \
  --equation heat \
  --dataset data/heat.npz \
  --output-dir runs/heat \
  --epochs 2000
```

训练损失包括：

- `pde_loss`: PDE 残差；
- `data_loss`: 解析数据拟合项；
- `boundary_loss`: 初始/边界条件残差。

输出包括：

- `model.pt`: 模型权重；
- `training_history.npy`: 训练历史；
- `config.json`: 训练配置。

## 实验 Pipeline

完整流程已经保存为 Python 脚本：

```bash
python scripts/run_experiment_pipeline.py --device cuda:1
```

该脚本会按顺序执行：

1. 生成/复用数据集；
2. 使用 improved 训练参数训练四类方程；
3. 评估并生成 `loss.png`、`fields.png`、`metrics.json`、`predictions.npz`。

也可以只运行某一个方程：

```bash
python scripts/run_experiment_pipeline.py --only heat --device cuda:1
```

当前 baseline pipeline 的训练参数相对最早 smoke/study 版本做了三类调整：

- 网络从 `32` 宽度提高到 `64`，层数增加；
- 训练 epoch 增加；
- 增加 `data_weight` 和 `boundary_weight`，线弹性降低 `pde_weight`，让场图像拟合更稳定。

默认输出目录不再使用 `improved` 后缀，而是：

```text
runs/baseline/heat
runs/baseline/burgers
runs/baseline/navier_stokes
runs/baseline/linear_elasticity
```

实验 pipeline 文件夹：

```text
experiments/baseline/run_pipeline.py
experiments/linear_elasticity_constitutive_ablation/run_ablation.py
experiments/heat_reduced_model_ablation/run_ablation.py
experiments/burgers_reduced_model_ablation/run_ablation.py
experiments/navier_stokes_reduced_model_ablation/run_ablation.py
experiments/linear_elasticity_reduced_model_ablation/run_ablation.py
```

线弹性 Hooke 本构残差消融实验：

```bash
python experiments/linear_elasticity_constitutive_ablation/run_ablation.py --device cuda:1
```

该脚本会比较：

- `without_constitutive`: 平衡方程 + 数据 + 边界，不加入 Hooke 本构残差；
- `with_constitutive`: 在上述基础上加入 Hooke 本构残差。

注意：Hooke 定律不是经验公式，而是线弹性 PDE 系统的本构闭合关系。如果网络同时输出位移和应力，完整的线弹性 PINN 物理残差应包含 Hooke 本构残差。

简化模型消融实验：

```bash
python experiments/heat_reduced_model_ablation/run_ablation.py --device cuda:1
python experiments/burgers_reduced_model_ablation/run_ablation.py --device cuda:1
python experiments/navier_stokes_reduced_model_ablation/run_ablation.py --device cuda:1
python experiments/linear_elasticity_reduced_model_ablation/run_ablation.py --device cuda:1
```

输出目录：

```text
runs/ablation/linear_elasticity_constitutive/
runs/ablation/<equation>_reduced_model/
```

## 数据集可视化

如果只看 `fields.png`，四个场景可能都会像二维热点图，因为当前数据集都是单位方形上的平滑制造解。可以用以下脚本直接查看数据集真值场：

```bash
python scripts/visualize_dataset.py --dataset data/heat.npz --out runs/dataset_preview/heat.png
python scripts/visualize_dataset.py --dataset data/burgers.npz --out runs/dataset_preview/burgers.png
python scripts/visualize_dataset.py --dataset data/navier_stokes.npz --out runs/dataset_preview/navier_stokes.png
python scripts/visualize_dataset.py --dataset data/linear_elasticity.npz --out runs/dataset_preview/linear_elasticity.png
```

该脚本会对正负变量使用发散色图，更容易看出速度、压力、应力、体力等不同物理量的符号和空间结构。

## 当前训练是否使用简化模型/对称性约束

目前已经完成的 `runs/*_study_gpu1` 训练没有加入简化模型软约束，也没有加入对称性软约束。它们使用的是 baseline PINN 损失：

```text
L = L_pde + L_data + L_boundary
```

唯一例外是 Navier-Stokes 使用了流函数 `psi`，这是一个结构性硬嵌入：速度由 `u=dpsi/dy, v=-dpsi/dx` 得到，因此不可压连续性 `u_x+v_y=0` 自动满足。

因此下一步非常适合做消融实验：

- baseline: `PDE + data + boundary`
- symmetry: baseline + `SymmetryResidual`
- reduced-model: baseline + `ReducedModelResidual`
- constitutive: 用于检查 Hooke 等本构关系是否作为完整 PDE 残差的一部分
- combined: baseline + symmetry + reduced-model
- hard-constraint: 使用 `MirrorFeatureMap`、`PeriodicFeatureMap` 或 `BoxDirichletTransform`

## 约束模块

`physics_embed/constraints.py` 提供第二阶段使用的模块：

- `SymmetryResidual`: 已知镜像、旋转、周期等对称性时，添加成对点残差；
- `ReducedModelResidual`: 将特定场景假设下的 PDE 简化模型写成类似 PDE 的残差；
- `BoxDirichletTransform`: 对矩形区域的 Dirichlet 边界条件使用硬约束输出变换。

这些模块可以在 PINN 训练循环中作为额外 loss 调用，也可以包装网络输出，在 PINN 前处理或后处理阶段使用。

## 参考来源

- PDEBench: `https://github.com/pdebench/PDEBench`
- PDEBench datasets: `https://doi.org/10.18419/darus-2986`
- 原始源码目录：`../Source_Code_orig/Source_Code` 与 `../Source_Code_0708/Source_Code`
