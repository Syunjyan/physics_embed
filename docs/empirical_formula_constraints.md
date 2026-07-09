# 四类方程经验公式与可选约束函数报告

## 1. 使用方式

经验公式不应无条件全域强加。建议先判断适用条件，再将其作为可选残差：

```python
from physics_embed.constraints import EmpiricalFormulaResidual
from physics_embed.empirical import navier_stokes_poiseuille

constraint = EmpiricalFormulaResidual(
    formula=lambda fields, points: navier_stokes_poiseuille(fields, points, centerline_velocity=1.0),
    weight=0.01,
)
loss = constraint(model, points, equation.prediction)
```

若公式只在边界、充分发展流段、远场或低 Reynolds 数区域成立，应只在对应点集上调用。

## 2. 热传导

### Fourier 定律

```text
q = -k grad(T)
```

适用条件：连续介质、局部热平衡、各向同性或已知导热系数张量。若网络同时预测温度和热流，可加入 `q+k grad(T)=0` 残差。

代码函数：`heat_fourier_law`。

### 对流换热边界

```text
-k grad(T).n = h(T-T_inf)
```

适用条件：边界与外界流体换热，换热系数 `h` 已知或可估计。适合作为 Robin 边界约束。

代码函数：`heat_robin_boundary`。

### 热阻公式

```text
Q = Delta T / R_th
```

适用条件：一维等效导热、层状材料、热路模型。更适合作为全局后处理残差，而非点态 PDE 残差。

代码函数：`heat_thermal_resistance`。

## 3. Burgers 方程

### 扩散主导近似

二维粘性 Burgers：

```text
u_t + u u_x + v u_y = nu Delta u
v_t + u v_x + v v_y = nu Delta v
```

当速度尺度小或黏性占优时，可近似为：

```text
u_t = nu Delta u
v_t = nu Delta v
```

代码函数：`burgers_diffusion_dominant`。

### 能量衰减趋势

无外部能量输入、黏性耗散占优时，动能不应持续增长。可把正增长部分作为弱惩罚：

```text
relu(d/dt (0.5*(u^2+v^2)))
```

代码函数：`burgers_energy_decay`。

注意：如果存在源项、边界输入或激波附近强非线性，该约束应降权或关闭。

## 4. Navier-Stokes

### Poiseuille 通道流

充分发展层流通道中：

```text
u(y) = 4 U_max y(H-y)/H^2
v = 0
```

适用条件：二维平行板、稳态、充分发展、层流、远离入口。适合用于通道流前处理预训练或出口段局部约束。

代码函数：`navier_stokes_poiseuille`。

### Bernoulli 近似

```text
p + 0.5 rho |u|^2 = const
```

适用条件：稳态、不可压、无粘近似、沿同一流线、无外力做功。对真实黏性流不可全域强制，可作为低权重远场/高速主流区约束。

代码函数：`navier_stokes_bernoulli`。

### Stokes 低 Reynolds 近似

低 Reynolds 数时忽略惯性项：

```text
grad(p) = nu Delta u
```

适用条件：爬行流、低 Reynolds 数。对一般高 Reynolds 流动不适用。

代码函数：`navier_stokes_stokes_limit`。

## 5. 线弹性

### Hooke 定律

小变形各向同性线弹性：

```text
epsilon = 0.5(grad(u)+grad(u)^T)
sigma = lambda tr(epsilon) I + 2 mu epsilon
```

适用条件：小变形、线弹性、各向同性。它是线弹性的核心本构关系，适合强残差或结构性嵌入。

代码函数：`linear_elasticity_hooke`。

### 单轴应力近似

```text
sigma_x = E epsilon_x
epsilon_y = -nu epsilon_x
sigma_y = 0
tau_xy = 0
```

适用条件：细长试样、单轴拉压、远离夹具和应力集中区域。适合做局部弱约束或 sanity check。

代码函数：`linear_elasticity_uniaxial_stress`。

## 6. 权重建议

- 本构类公式，如 Hooke、Fourier，若变量定义完全一致，可使用较高权重。
- 经验关联式，如 Bernoulli、Poiseuille、热阻公式，应先低权重使用。
- 全局公式尽量作为后处理指标；若进入训练，应对物理量先做无量纲化。
- 不确定适用条件时，先只输出残差图，不参与训练。

## 7. 已实现模块

实现文件：`physics_embed/empirical.py`

可选函数：

- `heat_fourier_law`
- `heat_robin_boundary`
- `heat_thermal_resistance`
- `burgers_diffusion_dominant`
- `burgers_energy_decay`
- `navier_stokes_poiseuille`
- `navier_stokes_bernoulli`
- `navier_stokes_stokes_limit`
- `linear_elasticity_hooke`
- `linear_elasticity_uniaxial_stress`
