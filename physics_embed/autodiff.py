import torch


def grad(output: torch.Tensor, inputs: torch.Tensor) -> torch.Tensor:
    """First derivative helper for PINN residuals."""
    return torch.autograd.grad(
        output,
        inputs,
        grad_outputs=torch.ones_like(output),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]


def column_grad(output: torch.Tensor, points: torch.Tensor, dim: int) -> torch.Tensor:
    """Derivative of a scalar output with respect to one coordinate column."""
    return grad(output, points)[:, dim : dim + 1]


def laplacian(output: torch.Tensor, points: torch.Tensor, dims=(0, 1)) -> torch.Tensor:
    """Laplacian over selected coordinate dimensions."""
    total = torch.zeros_like(output)
    for dim in dims:
        first = column_grad(output, points, dim)
        total = total + column_grad(first, points, dim)
    return total
