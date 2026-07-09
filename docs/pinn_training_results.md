# 四类方程 PINN 训练与评估结果

## 训练配置

所有正式结果均使用物理 1 号 GPU 训练：

```bash
CUDA_VISIBLE_DEVICES=1
```

训练目录：

- `runs/heat_study_gpu1`
- `runs/burgers_study_gpu1`
- `runs/navier_stokes_study_gpu1`
- `runs/linear_elasticity_study_gpu1`

每个目录包含：

- `model.pt`: 训练后的模型权重；
- `training_history.npy`: loss 历史；
- `loss.png`: loss 曲线；
- `fields.png`: 参考解、预测解、误差图；
- `metrics.json`: 相对 L2 误差和 MSE；
- `predictions.npz`: 预测场和参考场。

## 结果摘要

| 方程 | 主要物理量相对 L2 误差 | 训练 loss 变化 |
|---|---:|---:|
| Heat | `u=0.108` | `101.96 -> 0.356` |
| Burgers | `u=0.236`, `v=0.326` | `1.73 -> 0.157` |
| Navier-Stokes | `u=0.153`, `v=0.130`, `p=0.494` | `70.56 -> 1.13` |
| Linear Elasticity | `ux=0.501`, `uy=0.655`, `sigmaxx=0.203`, `sigmayy=0.371`, `sigmaxy=1.389` | `2234.79 -> 32.59` |

## 解释

热传导是最稳定的，误差已经降到约 10%。Burgers 的速度场趋势可学习，但 `v` 分量误差高于 `u`，继续训练或提高边界权重会改善。Navier-Stokes 使用流函数硬嵌入不可压条件，因此速度误差优于压力；压力缺少 gauge 固定和更强数据项，误差偏高是预期现象。线弹性包含位移、应力、平衡方程、本构关系和边界条件，损失尺度差异最大；当前轻量训练能明显下降，但要得到更高质量图像，需要增加 epoch、归一化各残差或使用自适应 loss 权重。

## 建议的下一步

1. 对线弹性增加无量纲化和分项 loss 权重，特别是平衡方程、本构方程、边界项之间的尺度平衡。
2. 对 Navier-Stokes 增加压力参考点或零均值约束，减少压力 gauge 不确定性。
3. 对 Burgers 和 NS 使用时间分阶段训练，先初值/边界，再 PDE 残差。
4. 对四类方程补 `--device` 或统一配置文件，防止后续误用 GPU。
