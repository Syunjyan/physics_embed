# 物理知识嵌入实验报告

## 1. 实验目的

TODO: 简述本实验的研究目标。

本实验围绕针对智能体的多层次，多领域知识嵌入展开，主要目标包括：

- 构建结构力学、热力学、流体力学方程的数据集生成与智能体学习流程；
- 验证物理公式、经验公式等领域知识作为软约束嵌入智能体的合理性；
- 验证对称性作为硬约束，在后处理阶段嵌入智能体的合理性；
- 验证数据知识在预训练阶段，通过迁移、插值等方法嵌入智能体的合理性；
- 通过消融实验比较不同知识模块对预测精度和物理一致性的影响。

## 2. 实验环境

TODO: 补充实际实验环境。

| 项目 | 配置 |
|---|---|
| 操作系统 | TODO |
| Python 环境 | TODO |
| PyTorch 版本 | TODO |
| GPU | TODO |
| CUDA | TODO |
| 代码路径 | `/home/houxinyun/PINNs/physics_embed` |

## 3. 数据集与物理场景

### 3.1 数据生成方法

TODO: 说明当前使用 manufactured solution method (MMS) 的原因、优点与限制。

当前数据集采用制造解方法生成。该方法先指定解析解，再由控制方程反推源项、体力项或边界条件，因此能够得到可验证的真值场和 PDE 残差。

### 3.2 热传导场景

TODO: 填写热传导实验描述。

| 项目 | 内容 |
|---|---|
| 方程类型 | 二维稳态热传导 |
| 区域 | `[0,1] x [0,1]` |
| 场变量 | `u`, `f` |
| 控制方程 | `-Delta u = f` |
| 解析解 | TODO |
| 边界条件 | TODO |
| 参数 | TODO |

### 3.3 Burgers 场景

TODO: 填写 Burgers 实验描述。

| 项目 | 内容 |
|---|---|
| 方程类型 | 二维粘性 Burgers |
| 区域 | `[0,1] x [0,1]`, `t in [0,1]` |
| 场变量 | `u`, `v`, `f_u`, `f_v` |
| 控制方程 | TODO |
| 解析解 | TODO |
| 参数 | `nu = TODO` |

### 3.4 Navier-Stokes 场景

TODO: 填写 Navier-Stokes 实验描述。

| 项目 | 内容 |
|---|---|
| 方程类型 | 二维不可压 Navier-Stokes |
| 区域 | `[0,1] x [0,1]`, `t in [0,1]` |
| 场变量 | `psi`, `p`, `u`, `v`, `f_u`, `f_v` |
| 控制方程 | TODO |
| 结构性先验 | 使用流函数 `psi` 自动满足不可压条件 |
| 参数 | `nu = TODO` |

### 3.5 线弹性场景

TODO: 填写线弹性实验描述。

| 项目 | 内容 |
|---|---|
| 方程类型 | 二维线弹性静力学 |
| 区域 | `[0,1] x [0,1]` |
| 场变量 | `ux`, `uy`, `sigmaxx`, `sigmayy`, `sigmaxy` |
| 控制方程 | `div(sigma)+f=0` |
| 几何方程 | TODO |
| 本构关系 | Hooke 定律 |
| 材料参数 | `lambda = TODO`, `mu = TODO` |

## 4. PINN 方法

### 4.1 网络结构

TODO: 描述 MLP 网络结构、激活函数、输入输出。

| 方程 | 输入 | 输出 | 网络结构 | 激活函数 |
|---|---|---|---|---|
| Heat | `(x,y)` | `u` | TODO | `tanh` |
| Burgers | `(x,y,t)` | `u,v` | TODO | `tanh` |
| Navier-Stokes | `(x,y,t)` | `psi,p` | TODO | `tanh` |
| Linear Elasticity | `(x,y)` | `ux,uy,sigmaxx,sigmayy,sigmaxy` | TODO | `tanh` |

### 4.2 损失函数

TODO: 描述 baseline loss。

Baseline PINN 的损失函数为：

```text
L = w_pde L_pde + w_data L_data + w_boundary L_boundary
```

其中：

- `L_pde`: PDE 残差损失；
- `L_data`: 数据拟合损失；
- `L_boundary`: 边界/初始条件损失。

### 4.3 物理公式嵌入


胡克定律嵌入

1.0系数

| Field | Metric | Without empirical | With empirical | Delta | Relative improvement |
|---|---:|---:|---:|---:|---:|
| ux | mse | 0.00092968 | 2.82385e-05 | -0.000901441 | 96.96% |
| ux | relative_l2 | 0.0610111 | 0.0106332 | -0.0503779 | 82.57% |
| uy | mse | 0.000344899 | 1.93912e-05 | -0.000325508 | 94.38% |
| uy | relative_l2 | 0.075798 | 0.0179727 | -0.0578253 | 76.29% |
| sigmaxx | mse | 0.00102251 | 0.00175552 | 0.000733012 | -71.69% |
| sigmaxx | relative_l2 | 0.00517052 | 0.00677492 | 0.00160439 | -31.03% |
| sigmayy | mse | 0.00119697 | 0.0012895 | 9.25311e-05 | -7.73% |
| sigmayy | relative_l2 | 0.00922921 | 0.0095793 | 0.00035009 | -3.79% |
| sigmaxy | mse | 0.000441459 | 0.00054833 | 0.000106871 | -24.21% |
| sigmaxy | relative_l2 | 0.0232926 | 0.0259593 | 0.00266675 | -11.45% |

经验公式软约束显著提升了位移场精度（ux/uy 相对误差下降约 83%/76%），但当前权重下对应力分量有轻微负面影响，说明 Hooke 约束有效但需要进一步调节应力数据项与经验公式项的权重。

0.1系数

| Field | Metric | Without empirical | With empirical | Delta | Relative improvement |
|---|---:|---:|---:|---:|---:|
| ux | mse | 0.00092968 | 0.000180959 | -0.000748721 | 80.54% |
| ux | relative_l2 | 0.0610111 | 0.0269173 | -0.0340938 | 55.88% |
| uy | mse | 0.000344899 | 0.000105847 | -0.000239053 | 69.31% |
| uy | relative_l2 | 0.075798 | 0.0419904 | -0.0338076 | 44.60% |
| sigmaxx | mse | 0.00102251 | 0.00126991 | 0.000247396 | -24.19% |
| sigmaxx | relative_l2 | 0.00517052 | 0.00576218 | 0.000591652 | -11.44% |
| sigmayy | mse | 0.00119697 | 0.000778925 | -0.000418049 | 34.93% |
| sigmayy | relative_l2 | 0.00922921 | 0.00744509 | -0.00178412 | 19.33% |
| sigmaxy | mse | 0.000441459 | 0.000420191 | -2.12679e-05 | 4.82% |
| sigmaxy | relative_l2 | 0.0232926 | 0.0227246 | -0.000568002 | 2.44% |

### 4.4 经验公式嵌入

TODO: 描述经验公式作为软约束的形式。

经验公式残差写为：

```text
L_emp = mean(|R_emp|^2)
```

总损失扩展为：

```text
L = w_pde L_pde + w_data L_data + w_boundary L_boundary + w_emp L_emp
```



本实验对比了在稀疏位移监督条件下，是否加入单轴 reduced-model 约束对二维线弹性 PINN 预测结果的影响。实验中 baseline 仅使用位移标签、平衡方程和边界条件，不使用内部应力标签，也不使用完整 plane-stress 本构残差；加入 reduced-model 后，额外施加

[
\sigma_{xx}=E\varepsilon_{xx},\quad
\sigma_{yy}=0,\quad
\sigma_{xy}=0
]

作为单轴应力场景下的简化物理先验。

结果显示，reduced-model 对**应力场预测的提升非常明显**。其中，(\sigma_{xx}) 的 MSE 从 (1.43\times10^{-4}) 降低到 (8.61\times10^{-6})，相对 (L_2) 误差从 0.598 降低到 0.147，分别提升约 **93.98%** 和 **75.47%**。横向应力 (\sigma_{yy}) 的 MSE 降低约 **95.16%**，相对 (L_2) 提升约 **78.00%**；剪应力 (\sigma_{xy}) 的 MSE 降低约 **98.48%**，相对 (L_2) 提升约 **87.68%**。这说明在没有应力监督和完整本构约束的情况下，单轴 reduced-model 能有效补充应力场信息，显著约束应力分量的物理形态。

位移场方面，(u_y) 也得到明显改善。其 MSE 从 (6.60\times10^{-7}) 降低到 (1.72\times10^{-7})，提升约 **74.02%**；相对 (L_2) 从 0.233 降低到 0.119，提升约 **49.03%**。这表明 reduced-model 不仅改善了应力推断，也通过单轴应力假设间接约束了横向位移响应，尤其是与泊松效应相关的横向变形。

不过，(u_x) 指标略有下降。加入 reduced-model 后，(u_x) 的 MSE 从 (1.94\times10^{-7}) 上升到 (2.29\times10^{-7})，相对 (L_2) 从 0.0379 上升到 0.0412，分别下降约 **18.12%** 和 **8.68%**。这个下降幅度相对较小，而且 (u_x) 原本已经是误差最低的场变量。较合理的解释是：reduced-model 约束主要作用于应力分量和横向应力/剪应力消除，对主拉伸方向位移 (u_x) 的直接帮助有限；同时新增约束改变了优化权重分配，使模型在极小的 (u_x) 误差上发生了轻微 trade-off。

因此，实验结论应表述为：

> 在 sparse-displacement-only 设置下，单轴 reduced-model 约束显著提升了应力场预测精度，并明显改善了横向位移 (u_y) 的预测；虽然主方向位移 (u_x) 出现轻微退化，但其误差本身已经较低，且退化幅度远小于应力场和 (u_y) 的收益。该结果说明 reduced-model 适合作为“稀疏监督 + 缺少应力标签 + 不使用完整本构残差”场景下的额外物理知识嵌入方式。


### 4.5 对称性嵌入

TODO: 描述当前对称性嵌入方式，以及后续可选方式。

可选方案包括：

- 成对点软残差；
- 硬对称坐标映射；
- 周期特征映射；
- Lie 点对称无穷小生成元残差；
- 群等变网络。

后处理硬约束不改网络参数，只在预测阶段投影：

q_sym(x,y) = 0.5 * (q(x,y) + parity * q(1-x,y))
变量奇偶性：

ux: -1          # 反对称
uy: +1          # 对称
sigmaxx: +1     # 对称
sigmayy: +1     # 对称
sigmaxy: -1     # 反对称
如果目标场确实满足镜像对称，而网络预测存在轻微不对称误差，投影后通常会降低误差。

## 5. 实验配置

TODO: 填写实际训练参数。

| 方程 | Epoch | Samples | Boundary Samples | Learning Rate | Loss Weights | Device |
|---|---:|---:|---:|---:|---|---|
| Heat | TODO | TODO | TODO | TODO | TODO | `cuda:1` |
| Burgers | TODO | TODO | TODO | TODO | TODO | `cuda:1` |
| Navier-Stokes | TODO | TODO | TODO | TODO | TODO | `cuda:1` |
| Linear Elasticity | TODO | TODO | TODO | TODO | TODO | `cuda:1` |

## 6. Baseline 实验结果

TODO: 填写 baseline 训练结果、loss 曲线和场图像路径。

| 方程 | 变量 | Relative L2 | MSE | 备注 |
|---|---|---:|---:|---|
| Heat | `u` | TODO | TODO | TODO |
| Burgers | `u` | TODO | TODO | TODO |
| Burgers | `v` | TODO | TODO | TODO |
| Navier-Stokes | `u` | TODO | TODO | TODO |
| Navier-Stokes | `v` | TODO | TODO | TODO |
| Navier-Stokes | `p` | TODO | TODO | TODO |
| Linear Elasticity | `ux` | TODO | TODO | TODO |
| Linear Elasticity | `uy` | TODO | TODO | TODO |
| Linear Elasticity | `sigmaxx` | TODO | TODO | TODO |
| Linear Elasticity | `sigmayy` | TODO | TODO | TODO |
| Linear Elasticity | `sigmaxy` | TODO | TODO | TODO |

## 7. 消融实验

### 7.1 实验设计

TODO: 描述消融组别。

| 实验组 | PDE Loss | Data Loss | Boundary Loss | Empirical Loss | Symmetry Loss |
|---|---|---|---|---|---|
| Baseline | Yes | Yes | Yes | No | No |
| + Empirical | Yes | Yes | Yes | Yes | No |
| + Symmetry | Yes | Yes | Yes | No | Yes |
| + Empirical + Symmetry | Yes | Yes | Yes | Yes | Yes |



本实验对比了在稀疏位移监督条件下，是否加入单轴 reduced-model 约束对二维线弹性 PINN 预测结果的影响。实验中 baseline 仅使用位移标签、平衡方程和边界条件，不使用内部应力标签，也不使用完整 plane-stress 本构残差；加入 reduced-model 后，额外施加

[
\sigma_{xx}=E\varepsilon_{xx},\quad
\sigma_{yy}=0,\quad
\sigma_{xy}=0
]

作为单轴应力场景下的简化物理先验。

结果显示，reduced-model 对**应力场预测的提升非常明显**。其中，(\sigma_{xx}) 的 MSE 从 (1.43\times10^{-4}) 降低到 (8.61\times10^{-6})，相对 (L_2) 误差从 0.598 降低到 0.147，分别提升约 **93.98%** 和 **75.47%**。横向应力 (\sigma_{yy}) 的 MSE 降低约 **95.16%**，相对 (L_2) 提升约 **78.00%**；剪应力 (\sigma_{xy}) 的 MSE 降低约 **98.48%**，相对 (L_2) 提升约 **87.68%**。这说明在没有应力监督和完整本构约束的情况下，单轴 reduced-model 能有效补充应力场信息，显著约束应力分量的物理形态。

位移场方面，(u_y) 也得到明显改善。其 MSE 从 (6.60\times10^{-7}) 降低到 (1.72\times10^{-7})，提升约 **74.02%**；相对 (L_2) 从 0.233 降低到 0.119，提升约 **49.03%**。这表明 reduced-model 不仅改善了应力推断，也通过单轴应力假设间接约束了横向位移响应，尤其是与泊松效应相关的横向变形。

不过，(u_x) 指标略有下降。加入 reduced-model 后，(u_x) 的 MSE 从 (1.94\times10^{-7}) 上升到 (2.29\times10^{-7})，相对 (L_2) 从 0.0379 上升到 0.0412，分别下降约 **18.12%** 和 **8.68%**。这个下降幅度相对较小，而且 (u_x) 原本已经是误差最低的场变量。较合理的解释是：reduced-model 约束主要作用于应力分量和横向应力/剪应力消除，对主拉伸方向位移 (u_x) 的直接帮助有限；同时新增约束改变了优化权重分配，使模型在极小的 (u_x) 误差上发生了轻微 trade-off。

因此，实验结论应表述为：

> 在 sparse-displacement-only 设置下，单轴 reduced-model 约束显著提升了应力场预测精度，并明显改善了横向位移 (u_y) 的预测；虽然主方向位移 (u_x) 出现轻微退化，但其误差本身已经较低，且退化幅度远小于应力场和 (u_y) 的收益。该结果说明 reduced-model 适合作为“稀疏监督 + 缺少应力标签 + 不使用完整本构残差”场景下的额外物理知识嵌入方式。

该实验验证了场景化 reduced-model 约束在稀疏监督条件下的有效性。与不使用 reduced-model 的 baseline 相比，加入单轴简化约束后，三个应力分量的误差均显著下降，其中 (\sigma_{xx})、(\sigma_{yy})、(\sigma_{xy}) 的相对 (L_2) 误差分别降低 75.47%、78.00% 和 87.68%。同时，横向位移 (u_y) 的相对 (L_2) 误差降低 49.03%。这说明在缺少内部应力标签和完整本构残差的情况下，reduced-model 能够提供有效的应力先验，并改善与单轴拉伸物理机制相关的位移响应。主方向位移 (u_x) 的误差略有增加，但其基线误差已经较低，且退化幅度明显小于应力场收益。整体而言，该实验支持将 reduced-model 作为稀疏监督场景下的低成本物理知识嵌入方式，而不是在全监督或完整本构约束已存在时重复加入。

| Field | Metric | Without reduced-model | With reduced-model | Delta | Relative improvement |
|---|---:|---:|---:|---:|---:|
| ux | mse | 1.93837e-07 | 2.28954e-07 | 3.51174e-08 | -18.12% |
| ux | relative_l2 | 0.0379272 | 0.0412199 | 0.0032927 | -8.68% |
| uy | mse | 6.60425e-07 | 1.71594e-07 | -4.88831e-07 | 74.02% |
| uy | relative_l2 | 0.233358 | 0.118949 | -0.114409 | 49.03% |
| sigmaxx | mse | 0.000143153 | 8.6144e-06 | -0.000134539 | 93.98% |
| sigmaxx | relative_l2 | 0.598234 | 0.146752 | -0.451482 | 75.47% |
| sigmayy | mse | 4.48652e-05 | 2.17214e-06 | -4.26931e-05 | 95.16% |
| sigmayy | relative_l2 | 0.321511 | 0.0707433 | -0.250768 | 78.00% |
| sigmaxy | mse | 6.02406e-05 | 9.14791e-07 | -5.93258e-05 | 98.48% |
| sigmaxy | relative_l2 | 0.372551 | 0.0459095 | -0.326642 | 87.68% |

### 7.2 线弹性经验公式消融

TODO: 填写线弹性 Hooke 经验公式消融结果。

| Field | Metric | Without Empirical | With Empirical | Delta | Relative Improvement |
|---|---:|---:|---:|---:|---:|
| `ux` | relative_l2 | TODO | TODO | TODO | TODO |
| `uy` | relative_l2 | TODO | TODO | TODO | TODO |
| `sigmaxx` | relative_l2 | TODO | TODO | TODO | TODO |
| `sigmayy` | relative_l2 | TODO | TODO | TODO | TODO |
| `sigmaxy` | relative_l2 | TODO | TODO | TODO | TODO |

### 7.3 其他方程消融

TODO: 填写 Heat、Burgers、Navier-Stokes 的消融实验结果。

### 7,4 对称性消融



| Field | Metric | Raw prediction | Symmetry projected | Delta | Relative improvement |
|---|---:|---:|---:|---:|---:|
| ux | mse | 6.43904e-06 | 1.27056e-06 | -5.16848e-06 | 80.27% |
| ux | relative_l2 | 0.66388 | 0.294901 | -0.368979 | 55.58% |
| uy | mse | 4.05312e-06 | 2.4384e-06 | -1.61472e-06 | 39.84% |
| uy | relative_l2 | 0.796313 | 0.617649 | -0.178664 | 22.44% |
| sigmaxx | mse | 0.000305776 | 0.000203441 | -0.000102334 | 33.47% |
| sigmaxx | relative_l2 | 0.330775 | 0.269805 | -0.0609694 | 18.43% |
| sigmayy | mse | 0.000134056 | 0.000109165 | -2.4891e-05 | 18.57% |
| sigmayy | relative_l2 | 0.374882 | 0.338293 | -0.0365889 | 9.76% |
| sigmaxy | mse | 3.18175e-05 | 2.22432e-05 | -9.57435e-06 | 30.09% |
| sigmaxy | relative_l2 | 0.767431 | 0.641659 | -0.125771 | 16.39% |

## 8. 结果分析

TODO: 分析各类物理场景下 PINN 的拟合表现。

建议讨论：

- 哪些方程更容易训练，为什么；
- 哪些变量误差较高，可能原因是什么；
- 经验公式软约束是否改善了主要物理量；
- 是否存在数据拟合与物理一致性之间的 trade-off；
- 对称性约束是否提升样本效率或泛化能力。

## 9. 可视化分析

TODO: 插入或引用图片路径。

| 方程 | Loss 曲线 | 场图像 | 数据集预览 |
|---|---|---|---|
| Heat | TODO | TODO | TODO |
| Burgers | TODO | TODO | TODO |
| Navier-Stokes | TODO | TODO | TODO |
| Linear Elasticity | TODO | TODO | TODO |

## 10. 局限性

TODO: 总结当前实验限制。

可能包括：

- 当前数据集是制造解，尚不代表真实工程复杂几何；
- 四个场景都在单位方形区域，视觉差异有限；
- 当前经验公式权重尚未系统调参；
- 消融实验尚需多随机种子重复；
- 未统计训练时间、显存、推理时间等效率指标。

## 11. 后续工作

TODO: 填写下一步计划。

建议方向：

- 引入真实 PDEBench 或 FEM/CFD 数据；
- 增加多随机种子实验；
- 对经验公式权重做网格搜索；
- 加入对称性消融实验；
- 使用硬边界约束和 hard-soft 混合 PINN；
- 增加训练效率和显存指标。

## 12. 附录

### 12.1 运行命令

TODO: 粘贴实际命令。

```bash
# baseline
python scripts/run_experiment_pipeline.py --device cuda:1

# linear elasticity empirical ablation
python experiments/linear_elasticity_empirical_ablation/run_ablation.py --device cuda:1
```

### 12.2 代码路径

```text
physics_embed/
├── physics_embed/
├── scripts/
├── experiments/
├── docs/
├── data/
└── runs/
```

