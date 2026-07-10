# Navier-Stokes 黏度插值预训练实验

## 1. 实验目的

本实验验证：对于同一类 Navier-Stokes 场景，仅运动黏度 `nu` 不同的两个场模型，能否通过网络权重插值得到第三个黏度的初始化，并在目标黏度上更快或更好地训练。

对比组：

- `target_nu_c_scratch`: 在 `nu=c` 数据集上从随机初始化开始训练；
- `target_nu_c_interpolated`: 先训练 `nu=a` 和 `nu=b` 两个模型，再对其对齐权重线性插值，作为 `nu=c` 的初始化。

## 2. 数据生成

PDEBench 的 2D incompressible Navier-Stokes 数据由数值求解器生成，并以 `[N, T, X, Y, V]` 结构化场数据保存。本实验**不复现 PDEBench 的数值求解流程**。

本实验采用 **Taylor-Green vortex 解析解**（`physics_embed/ns_taylor_green.py`）生成可精确验证的数据：

- 周期边界、规则时空网格；
- 保存与 PDEBench 兼容的张量布局 `fields_st`（shape `[1, T, H, W, 3]`，变量顺序 `u, v, p`）；
- 同时导出场学习所需的坐标点云 `points` 与逐点场值；
- 数据值直接来自解析解，不引入数值离散误差。

解析解形式：

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
- 可直接验证数据、方程和参数的一致性；
- 不需要大型数值求解依赖，适合隔离研究黏度变化下的参数迁移。

## 3. 实验配置

默认参数：

| 参数 | 值 |
|---|---:|
| `nu_a` | `0.005` |
| `nu_b` | `0.02` |
| `nu_c` | `0.01` |
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

其中 `theta_a` 和 `theta_b` 分别是 `nu=a`、`nu=b` 的训练后模型权重。两个源模型必须从**同一组初始权重**出发，并使用相同网络结构和采样序列，以维持隐藏单元对齐；独立随机初始化模型不能直接进行有意义的逐参数插值。目标对照组也从同一组公共初始权重出发。

## 5. 目录结构

与仓库其他实验保持一致：数据放在 `data/`，训练产物放在 `runs/ablation/`。

```text
physics_embed/
├── data/
│   ├── ns_tg_nu_0.005.npz
│   ├── ns_tg_nu_0.01.npz
│   └── ns_tg_nu_0.02.npz
└── runs/ablation/ns_viscosity_interpolation/
    ├── source_nu_a/
    ├── source_nu_b/
    ├── target_nu_c_scratch/
    ├── target_nu_c_interpolated/
    ├── metrics_table.md
    ├── summary.json
    └── experiment_config.json
```

主要输出：

- `data/ns_tg_nu_*.npz`: 三个黏度的数据集；
- `source_nu_a/`, `source_nu_b/`: 源黏度模型；
- `target_nu_c_scratch/`: 目标黏度从头训练；
- `target_nu_c_interpolated/`: 目标黏度插值初始化训练；
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
  --data-root /tmp/ns_viscosity_interpolation_smoke_data \
  --output-root runs/ablation/ns_viscosity_interpolation_smoke \
  --spatial-resolution 8 \
  --time-steps 5 \
  --hidden 16 16 \
  --source-epochs 2 \
  --target-epochs 2 \
  --samples 64 \
  --boundary-samples 32
```

## 7. 实验结果

共享初始权重、对齐源模型并完成 600 轮目标训练后：

| 场 | Scratch relative L2 | Interpolated relative L2 | 相对改善 |
|---|---:|---:|---:|
| `u` | 0.4211 | 0.0923 | 78.09% |
| `v` | 0.4205 | 0.0828 | 80.32% |
| `p` | 0.9898 | 0.2052 | 79.27% |
| 三场均值 | 0.6105 | 0.1267 | 79.24% |

训练前均值也由 1.2259 降至 0.9910。该结果支持黏度区间内、表示对齐条件下的参数迁移，但当前只有一个随机种子，且评估为同一网格上的场重建。

进一步改进方向：

- 使用更接近的 `nu_a, nu_b`；
- 对压力增加 gauge 约束；
- 先插值再只微调部分层；
- 使用参数化场模型，把 `nu` 作为输入。

