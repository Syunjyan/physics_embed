"""Physics embedding experiments for 2D PDE datasets and PINNs."""

from physics_embed.equations import EQUATIONS, get_equation
from physics_embed.models import MLP

__all__ = ["EQUATIONS", "MLP", "get_equation"]

