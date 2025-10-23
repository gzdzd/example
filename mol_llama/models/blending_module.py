from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from .atom_moe import AtomMoE


class BlendingModule(nn.Module):
    """High level module that blends 2D and 3D molecular features.

    The original project used a bespoke ``blending_module`` implemented
    on top of a collection of feed-forward networks.  The implementation
    in :mod:`mol_llama.models.atom_moe` provides a more expressive
    Mixture-of-Experts (MoE) router tailored for atom-level features.

    This class wraps :class:`~mol_llama.models.atom_moe.AtomMoE` and keeps
    the external API similar to the former blending module so that other
    parts of the project can instantiate the blender without code
    changes.  It exposes a ``forward`` method returning the fused hidden
    states and, optionally, routing statistics.
    """

    def __init__(
        self,
        d2d: int,
        d3d: int,
        df: int = 256,
        n_experts: int = 6,
        topk: int = 2,
        gate_temp: float = 1.2,
        gate_noise: float = 0.0,
        dropout: float = 0.1,
        capacity_factor: float = 1.25,
        use_gumbel: bool = False,
        use_hard_capacity: bool = True,
        out_dim: Optional[int] = None,
        always_return_stats: bool = False,
        detach_stats: bool = True,
        atom_moe_cls: type[AtomMoE] = AtomMoE,
        atom_moe_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        atom_moe_kwargs = atom_moe_kwargs or {}
        self.moe = atom_moe_cls(
            d2d=d2d,
            d3d=d3d,
            df=df,
            n_experts=n_experts,
            topk=topk,
            gate_temp=gate_temp,
            gate_noise=gate_noise,
            dropout=dropout,
            capacity_factor=capacity_factor,
            use_gumbel=use_gumbel,
            use_hard_capacity=use_hard_capacity,
            **atom_moe_kwargs,
        )

        self.input_dims = (d2d, d3d)
        self.hidden_dim = df
        self.output_dim = df if out_dim is None else out_dim
        if self.output_dim == df:
            self.output_proj: nn.Module = nn.Identity()
        else:
            self.output_proj = nn.Linear(df, self.output_dim)

        self.post_dropout: nn.Module
        if dropout > 0:
            self.post_dropout = nn.Dropout(dropout)
        else:
            self.post_dropout = nn.Identity()

        self.always_return_stats = always_return_stats
        self.detach_stats = detach_stats
        self._last_stats: Optional[Dict[str, torch.Tensor]] = None

    def forward(
        self,
        h2d: torch.Tensor,
        h3d: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_stats: bool = False,
        **unused: Any,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Fuse the 2D and 3D features.

        Args:
            h2d: 2D features with shape ``[batch, atoms, d2d]``.
            h3d: 3D features with shape ``[batch, atoms, d3d]``.
            mask: Optional boolean mask identifying the valid atoms.
            return_stats: Whether to return routing statistics.

        Returns:
            The fused features.  When ``return_stats`` is ``True`` the
            method returns a tuple ``(features, stats)`` where
            ``stats`` contains the router metrics reported by
            :class:`AtomMoE`.
        """

        # ``unused`` keeps backwards compatibility with call-sites that
        # might have provided legacy keyword arguments.
        if unused:
            # No-op: arguments are intentionally ignored.  The branch is
            # kept explicit so that future developers notice the
            # behaviour when reading the source.
            pass

        needs_stats = return_stats or self.always_return_stats
        fused, stats = self.moe(h2d, h3d, mask=mask, return_stats=needs_stats)
        fused = self.output_proj(fused)
        fused = self.post_dropout(fused)

        if needs_stats:
            if stats is None:
                stats = {}
            if self.detach_stats:
                stats = {
                    name: value.detach() if isinstance(value, torch.Tensor) else value
                    for name, value in stats.items()
                }
            self._last_stats = stats
            if return_stats:
                return fused, stats
        else:
            self._last_stats = None
        return fused

    def forward_with_stats(
        self,
        h2d: torch.Tensor,
        h3d: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Convenience wrapper returning both fused states and stats."""

        return self.forward(h2d, h3d, mask=mask, return_stats=True)

    def get_last_router_stats(self, clear: bool = False) -> Optional[Dict[str, torch.Tensor]]:
        """Return the router statistics collected during the last call."""

        stats = self._last_stats
        if clear:
            self._last_stats = None
        return stats


class AtomMoEBlendingModule(BlendingModule):
    """Alias maintained for backwards compatibility."""

    pass


def build_blending_module(**kwargs: Any) -> BlendingModule:
    """Factory helper compatible with older configuration files."""

    if "d2d" not in kwargs or "d3d" not in kwargs:
        missing = {key for key in ("d2d", "d3d") if key not in kwargs}
        raise ValueError(f"build_blending_module requires {missing} to be provided")
    return BlendingModule(**kwargs)


__all__ = [
    "BlendingModule",
    "AtomMoEBlendingModule",
    "build_blending_module",
]
