# Navier-Stokes 黏度插值预训练实验

## 1. 实验目的

本实验验证：对于同一类 Navier-Stokes 场景，仅运动黏度 `nu` 不同的两个 PINN，能否通过网络权重插值得到第三个黏度的初始化，并在目标黏度上更快或更好地训练。

对比组：

- `target_nu_c_scratch`: 在 `nu=c` 数据集上从随机初始化开始训练；
- `target_nu_c_interpolated`: 先训练 `nu=a` 和 `nu=b` 两个 PINN，再对其权重线性插值，作为 `nu=c` 的初始化。

## 2. 数据生成

PDEBench 的 2D incompressible Navier-Stokes 数据由数值求解器（Phiflow：semi-Lagrangian 平流 + explicit diffusion + pressure projection）生成，并以 `[N, T, X, Y, V]` 结构化场数据保存。

本实验采用 **Taylor-Green vortex** 作为基准场景，并使用 **pseudo-spectral projection 数值求解器**（`physics_embed/ns_spectral.py`）生成数据，尽量贴近 PDEBench 的“数值求解 + 结构化时空场”流程：

- 周期边界、结构化网格、显式时间推进；
- 保存 PDEBench 风格张量 `fields_st`（shape `[1, T, H, W, 3]`，变量顺序 `u, v, p`）；
- 同时导出 PINN 训练所需的 collocation 点云 `points` 与逐点场值；
- 每帧与 Taylor-Green 解析解对比，记录 `solver_max_abs_error` 作为数值验证。

解析解形式（用于验证，而非直接填数据）：

```text
u = -cos(kx) sin(ky) exp(-2 nu k^2 t)
v =  sin(kx) cos(ky) exp(-2 nu k^2 t)
p = -1/4 [cos(2kx)+cos(2ky)] exp(-4 nu k^2 t)
k = 2*pi
```

该场景满足二维不可压 Navier-Stokes：

```text
u_t + u u_x + v u_y + p_x - nu Delta u = 0
v_t + u v_x + v v_y + p_y - nu Delta v = 0
u_x + v_y = 0
```

选择 Taylor-Green vortex 的原因：

- 它是不可压 NS 的标准解析基准；
- 黏度 `nu` 显式控制速度场随时间衰减；
- 谱方法数值解可与解析解逐帧对照，验证数据生成正确性；
- 不需要 Phiflow/JAX 等大型依赖，便于快速验证插值预训练思想。

## 3. 实验配置

默认参数：

| 参数 | 值 |
|---|---:|
| `nu_a` | `0.005` |
| `nu_b` | `0.02` |
| `nu_c` | `0.0125` |
| spatial resolution | `32 x 32` |
| time steps | `21` |
| hidden layers | `64,64,64,64` |
| source epochs | `1500` |
| target epochs | `800` |
| device | `cuda:1` |

## 4. 权重插值

当 `c` 位于 `a` 和 `b` 之间时：

```text
alpha = (c-a)/(b-a)
theta_c_init = (1-alpha) theta_a + alpha theta_b
```

其中 `theta_a` 和 `theta_b` 分别是 `nu=a`、`nu=b` 的训练后 PINN 权重。

## 5. 输出

实验目录：

```text
runs/ablation/ns_viscosity_interpolation/
```

主要输出：

- `data/ns_tg_nu_*.npz`: 三个黏度的数据集；
- `runs/source_nu_a/`, `runs/source_nu_b/`: 源黏度模型；
- `runs/target_nu_c_scratch/`: 目标黏度从头训练；
- `runs/target_nu_c_interpolated/`: 目标黏度插值初始化训练；
- `metrics_table.md`: scratch vs interpolation 指标表（含训练前 init 与训练后 final）；
- `summary.json`: 聚合指标与插值系数；
- `gt_pred_speed.gif`: 每个 target run 的 GT/Pred/Error 速度模长动图。

## 6. 运行命令

```bash
cd /home/houxinyun/PINNs/physics_embed

/home/houxinyun/miniconda3/envs/gensound/bin/python \
  experiments/navier_stokes_viscosity_interpolation/run_experiment.py \
  --device cuda:1 \
  --output-root runs/ablation/ns_viscosity_interpolation
```

快速 smoke test：

```bash
/home/houxinyun/miniconda3/envs/gensound/bin/python \
  experiments/navier_stokes_viscosity_interpolation/run_experiment.py \
  --device cuda:1 \
  --output-root runs/ablation/ns_viscosity_interpolation_smoke \
  --spatial-resolution 8 \
  --time-steps 5 \
  --hidden 16 16 \
  --source-epochs 2 \
  --target-epochs 2 \
  --samples 64 \
  --boundary-samples 32
```

## 7. 结果解读

如果 `target_nu_c_interpolated` 在相同 target epochs 下：

- `relative_l2` 更低；
- `mse` 更低；
- loss 下降更快；
- GIF 中误差更小；

则说明黏度参数相近场景之间的 PINN 权重插值预训练有效。

如果只有部分变量改善，通常说明插值初始化对某些物理量更敏感，例如速度场改善但压力场不稳定。若整体变差，则可能需要：

- 使用更接近的 `nu_a, nu_b`；
- 对压力增加 gauge 约束；
- 先插值再只微调部分层；
- 使用参数化 PINN，把 `nu` 作为输入。

