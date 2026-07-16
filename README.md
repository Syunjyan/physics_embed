# physics_embed

`physics_embed` 是一个面向二维物理方程的实验仓库。

## 当前方程

- `heat`: 二维稳态热传导，形式为 `-Delta u = f`。
- `burgers`: 二维粘性 Burgers 方程，输出速度分量 `u, v`。
- `navier_stokes`: 二维不可压 Navier-Stokes 方程，网络输出流函数 `psi` 和压力 `p`，速度由自动微分得到。
- `linear_elasticity`: 二维线弹性静力学，输出位移和应力，残差包含平衡方程、几何方程和本构关系。

当前数据集采用 manufactured solution method (MMS, 制造解方法) 生成。这样每个方程都有解析真值、源项或体力项、边界条件，便于验证数据生成和 PINN 残差实现。

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
python scripts/generate_dataset.py --equation linear_elasticity_mirror --out data/linear_elasticity_mirror.npz
python scripts/generate_ns_taylor_green.py --viscosity 0.01
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

正式实验入口：

```text
experiments/baseline/run_pipeline.py
experiments/linear_elasticity_constitutive_ablation/run_ablation.py
experiments/linear_elasticity_uniaxial_sparse_reduced_model_ablation/run_ablation.py
experiments/linear_elasticity_mirror_postprocess_symmetry/run_ablation.py
experiments/navier_stokes_viscosity_interpolation/run_experiment.py
```

线弹性 Hooke 本构残差消融实验：

```bash
python experiments/linear_elasticity_constitutive_ablation/run_ablation.py --device cuda:1
```

该脚本会比较：

- `without_constitutive`: 平衡方程 + 数据 + 边界，不加入 Hooke 本构残差；
- `with_constitutive`: 在上述基础上加入 Hooke 本构残差。

注意：Hooke 定律不是经验公式，而是线弹性 PDE 系统的本构闭合关系。如果网络同时输出位移和应力，完整的线弹性 PINN 物理残差应包含 Hooke 本构残差。

场景匹配的简化模型、预测后对称投影和参数迁移实验：

```bash
python experiments/linear_elasticity_uniaxial_sparse_reduced_model_ablation/run_ablation.py --device cuda:1
python experiments/linear_elasticity_mirror_postprocess_symmetry/run_ablation.py --device cuda:1
python experiments/navier_stokes_viscosity_interpolation/run_experiment.py --device cuda:1
```

注意：`heat`、`burgers`、通用 `navier_stokes` 的 reduced-model 入口所使用的默认简化假设与当前制造解场景不匹配，只能作为假设失配负对照，不能作为正向结论。

正式报告与审计：

```text
docs/experiment_report_template.md
docs/experiment_rigor_audit.md
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

## 实验严谨性说明

- 当前指标来自同一规则网格上的场重建，不是独立测试集泛化指标；
- 正式消融均为单随机种子；
- 简化模型必须先验证适用假设；
- Taylor-Green 数据为解析解，使用 PDEBench 兼容布局但不复现其数值求解器；
- 权重插值要求两个源模型共享初始权重以保持隐藏单元对齐。

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
