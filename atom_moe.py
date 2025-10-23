"""Compatibility wrapper for downstream imports.

The original repository exposed :class:`AtomMoE` at the project root.
To keep that behaviour while moving the implementation into the
``mol_llama`` package we re-export the public API here.
"""

from mol_llama.models.atom_moe import AtomMoE, ExpertFFN

__all__ = ["AtomMoE", "ExpertFFN"]
