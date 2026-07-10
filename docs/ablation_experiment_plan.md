# 消融实验结构

## 1. 概念澄清

本项目后续区分三类约束：

- **完整 PDE / 物理残差**：控制方程、几何方程、本构关系、边界条件。例如线弹性中的 Hooke 定律属于本构关系，是完整线弹性 PINN 残差的一部分。
- **简化模型约束**：原 PDE 在特定场景假设下得到的近似模型，例如 Stokes 极限、Poiseuille 通道流、Burgers 扩散主导近似、1D 稳态导热近似。
- **对称性约束**：几何、边界、载荷或初值导致的镜像、周期、旋转或 Lie 点对称。

因此，旧的 `empirical` 命名已移除。Hooke/Fourier/Robin/能量衰减不再作为“经验公式消融”。

## 2. 实验目录

```text
experiments/
├── baseline/
│   └── run_pipeline.py
├── heat_reduced_model_ablation/
│   └── run_ablation.py
├── burgers_reduced_model_ablation/
│   └── run_ablation.py
├── navier_stokes_reduced_model_ablation/
│   └── run_ablation.py
├── linear_elasticity_reduced_model_ablation/
│   └── run_ablation.py
└── linear_elasticity_constitutive_ablation/
    └── run_ablation.py
```

输出目录：

```text
runs/baseline/<equation>/
runs/ablation/<equation>_reduced_model/
runs/ablation/linear_elasticity_constitutive/
```

## 3. Baseline

Baseline 使用：

```text
L = w_pde L_pde + w_data L_data + w_boundary L_boundary
```

当前 baseline 没有加入简化模型软约束，也没有加入对称性软约束。Navier-Stokes 的流函数属于结构性硬约束。

## 4. 线弹性本构消融

脚本：

```bash
python experiments/linear_elasticity_constitutive_ablation/run_ablation.py --device cuda:1
```

对比组：

- `without_constitutive`: 平衡方程 + 数据 + 边界，不加入 Hooke 本构残差；
- `with_constitutive`: 在上述基础上加入 Hooke 本构残差。

这个实验回答的是：当网络同时输出位移和应力时，Hooke 本构关系是否必须作为完整 PINN 物理残差的一部分。

## 5. 简化模型消融

脚本：

```bash
python experiments/heat_reduced_model_ablation/run_ablation.py --device cuda:1
python experiments/burgers_reduced_model_ablation/run_ablation.py --device cuda:1
python experiments/navier_stokes_reduced_model_ablation/run_ablation.py --device cuda:1
python experiments/linear_elasticity_reduced_model_ablation/run_ablation.py --device cuda:1
```

对比组：

- `without_reduced_model`: baseline；
- `with_reduced_model`: baseline + 特定假设下的简化模型残差。

注意：简化模型只有在场景假设成立时才应该改善结果。如果当前 MMS 数据不满足该假设，加入简化模型可能使误差变差；这本身也是有效的消融结论。

## 6. 汇总表

使用：

```bash
python scripts/summarize_ablation.py \
  --output-root runs/ablation/<name> \
  --without-name without_reduced_model \
  --with-name with_reduced_model \
  --label reduced-model
```

线弹性本构消融：

```bash
python scripts/summarize_ablation.py \
  --output-root runs/ablation/linear_elasticity_constitutive \
  --without-name without_constitutive \
  --with-name with_constitutive \
  --label constitutive
```

