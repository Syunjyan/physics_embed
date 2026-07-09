# 消融实验结构与线弹性经验公式实验

## 实验目录结构

当前将实验 pipeline 放在 `experiments/` 下：

```text
experiments/
├── baseline/
│   └── run_pipeline.py
└── linear_elasticity_empirical_ablation/
    └── run_ablation.py
```

输出放在 `runs/` 下：

```text
runs/baseline/<equation>/
runs/ablation/linear_elasticity_empirical/
```

## Baseline

Baseline 使用：

```text
L = w_pde L_pde + w_data L_data + w_boundary L_boundary
```

当前 baseline 没有加入对称性软约束，也没有加入经验公式软约束。Navier-Stokes 的流函数属于结构性硬约束，不属于消融中的 soft empirical/symmetry loss。

## 线弹性经验公式消融

脚本：

```bash
python experiments/linear_elasticity_empirical_ablation/run_ablation.py --device cuda:1
```

对比组：

1. `without_empirical`

   使用静力平衡方程、数据拟合和边界条件，但故意不加入本构/Hooke 残差。

2. `with_hooke_empirical`

   在 `without_empirical` 基础上加入 Hooke 定律残差：

   ```text
   sigma = lambda tr(epsilon) I + 2 mu epsilon
   ```

这个设计用于回答：当应力和位移都由网络输出时，Hooke 经验/本构公式作为软约束是否能提升应力-位移一致性与泛化。

## 评价指标

每个 run 会输出：

- `metrics.json`: 位移、应力的相对 L2 误差；
- `fields.png`: 真值、预测、误差图；
- `loss.png`: 单个 run 的 loss；
- `ablation_loss.png`: 两组加权 loss 对比；
- `summary.json`: 两组最终误差与 empirical loss 汇总。

## 注意事项

短 epoch smoke test 只验证代码可运行，不用于判断经验公式是否有效。正式比较应使用默认参数，或更长训练：

```bash
python experiments/linear_elasticity_empirical_ablation/run_ablation.py \
  --device cuda:1 \
  --epochs 3000
```

如果 `with_hooke_empirical` 的应力误差、Hooke residual 和边界误差同时更低，说明经验公式软约束有效。若只降低 Hooke residual 但数据误差变差，说明权重过高或公式适用范围需要限制。
