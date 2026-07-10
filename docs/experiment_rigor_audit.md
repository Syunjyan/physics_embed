# 实验严谨性审计

审计日期：2026-07-10

## 1. 审计范围

审计覆盖 `experiments/` 下全部入口、`physics_embed/` 中的数据生成与约束实现，以及当前可复现的消融结果。检查项包括：

- 物理假设是否与数据场景一致；
- 对照组是否只改变一个因素；
- 初始化、随机种子、训练预算和监督信息是否公平；
- 指标是否来自声明的数据；
- 结果是否足以支撑报告中的结论；
- 数据、配置、模型和可视化是否可再生成。

可执行的数据一致性检查位于 `scripts/validate_experiment_assets.py`。当前九个正式数据集均通过有限值、解析场一致性、结构化数组一致性检查；单轴简化模型最大假设残差为 `0`，镜像奇偶性最大误差为 `2.05e-8`。

## 2. 审计结论

### 2.1 可作为正式证据的实验

1. **Hooke 本构闭合消融**
   - 对照组使用相同初始化、随机种子、训练预算和完整场监督；
   - 唯一变量是是否加入本构残差；
   - 结果可用于说明多目标约束的收益与权衡，不能用于证明泛化。

2. **稀疏单轴简化模型消融**
   - 数据严格满足单轴应力假设；
   - 两组均只使用内部位移标签、平衡方程和边界条件，不使用内部应力标签；
   - 唯一变量是是否加入单轴简化关系；
   - 结果可用于说明在假设成立且应力监督缺失时的应力恢复收益。

3. **镜像对称后处理**
   - 数据奇偶性已数值验证；
   - 原始预测与投影预测来自同一模型；
   - 结果可用于说明正交投影能够去除违反已知对称性的分量；
   - 该实验不是训练期对称约束消融，且在对称真值和成对网格上，平方误差不增具有数学保证。

4. **黏度参数族的对齐权重插值**
   - 两个源模型使用相同网络结构、公共初始权重和采样序列，避免隐藏单元置换导致的无意义插值；
   - 目标 scratch 组从同一公共初始权重出发；
   - 两个目标组训练预算和采样种子相同；
   - 评估直接使用目标数据集中的 `u,v,p`，不再重新生成另一套真值。

### 2.2 仅作为工程基线或方法实现

- `experiments/baseline/`：可验证训练管线能运行，但训练与评估使用同一规则网格，不是独立测试集结果。
- 通用物理残差、边界条件、流函数表示：属于实现基础，不作为单独效果结论。
- `physics_embed/constraints.py` 中尚未进入正式消融的约束：只可描述为已实现或候选方法。

### 2.3 不进入正式结论的实验

- `heat_reduced_model_ablation`：默认一维无源稳态导热假设与当前二维有源制造解不一致。
- `burgers_reduced_model_ablation`：扩散主导近似与当前含非线性平流和制造源项的数据不一致。
- `navier_stokes_reduced_model_ablation`：Stokes 极限与当前非定常制造解未建立低 Reynolds 数条件。
- `linear_elasticity_reduced_model_ablation`：全场监督与完整本构残差已提供同类信息，额外简化关系主要测试冗余而非缺失知识补充。
- 历史 `*_empirical` 结果：术语和脚本均已废弃，不作为当前代码的可复现实验证据。

这些入口保留为**假设失配负对照或后续专用场景实验的脚手架**，报告不得将其解释为通用正向结果。

## 3. 仍然存在的统计与外推限制

1. 正式消融目前均为单随机种子，不能给出显著性结论或置信区间。
2. 训练点来自完整规则网格的重复随机抽样，评估仍在同一网格上；本文指标应称为“场重建误差”，而不是测试集泛化误差。
3. 数据主要来自制造解或解析解，适合因果隔离和代码验证，不代表复杂几何、噪声测量或真实 CFD/FEM 数据。
4. 各实验只验证了假设匹配的内插场景；未验证参数外推、简化模型失配和对称性破缺。
5. 未系统统计训练时间、显存、推理时间和不同约束权重的敏感性。

因此，报告中的结论统一使用“在当前受控场景和单次运行中观察到”，不使用“普遍有效”“显著提升”或“等价于真实工程数据”等过度表述。

## 4. 复现检查

```bash
python scripts/validate_experiment_assets.py

CUDA_VISIBLE_DEVICES=1 python \
  experiments/linear_elasticity_constitutive_ablation/run_ablation.py \
  --device cuda:0

CUDA_VISIBLE_DEVICES=1 python \
  experiments/linear_elasticity_uniaxial_sparse_reduced_model_ablation/run_ablation.py \
  --device cuda:0

CUDA_VISIBLE_DEVICES=1 python \
  experiments/linear_elasticity_mirror_postprocess_symmetry/run_ablation.py \
  --device cuda:0

CUDA_VISIBLE_DEVICES=1 python \
  experiments/navier_stokes_viscosity_interpolation/run_experiment.py \
  --device cuda:0
```
