import torch, torch.nn as nn, torch.nn.functional as F

class ExpertFFN(nn.Module):
    def __init__(self, df: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(df),
            nn.Linear(df, expansion * df),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * df, df),
        )

    def forward(self, x):
        return self.net(x)

class AtomMoE(nn.Module):
    """Atom-level MoE upgraded with DeepSeek-style features:
    - Stronger gate (LN+MLP) with optional Gumbel noise
    - Hard Top-1 routing with per-expert capacity (Switch-like)
    - Fallback to Top-k sparse weighting for k>1
    - Router z-loss and load/importance stats
    - Residual fusion on shared base
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
    ):
        super().__init__()
        self.din = d2d + d3d
        self.df = df
        self.K = n_experts
        self.topk = max(1, min(topk, n_experts))
        self.gate_temp = gate_temp
        self.gate_noise = gate_noise
        self.capacity_factor = capacity_factor
        self.use_gumbel = use_gumbel
        self.use_hard_capacity = use_hard_capacity

        self.input_proj = nn.Linear(self.din, df)
        # Stronger gate: LN + MLP
        self.gate_mlp = nn.Sequential(
            nn.LayerNorm(df),
            nn.Linear(df, max(df // 2, 64)),
            nn.GELU(),
            nn.Linear(max(df // 2, 64), n_experts),
        )
        # Shared base feeding experts
        self.shared = nn.Sequential(nn.LayerNorm(df), nn.GELU(), nn.Linear(df, df))
        # Expert FFNs
        self.expert_heads = nn.ModuleList(
            [ExpertFFN(df, expansion=4, dropout=dropout) for _ in range(n_experts)]
        )
        self.out_proj = nn.Sequential(nn.Dropout(dropout), nn.Linear(df, df))

    def _add_noise(self, logits):
        if self.gate_noise > 0:
            logits = logits + self.gate_noise * torch.randn_like(logits)
        if self.use_gumbel:
            eps = 1e-9
            u = torch.rand_like(logits)
            g = -torch.log(-torch.log(u + eps) + eps)
            logits = logits + g
        return logits

    def _mask_logits(self, logits, mask):
        if mask is None:
            return logits
        minus_inf = torch.tensor(-1e9, device=logits.device, dtype=logits.dtype)
        return logits.masked_fill((~mask.bool()).unsqueeze(-1), minus_inf)

    def _router_zloss(self, logits):
        z = torch.logsumexp(logits, dim=-1)
        return (z ** 2).mean()

    def _topk_soft(self, logits):
        scores = logits / self.gate_temp
        vals, idx = torch.topk(scores, k=self.topk, dim=-1)
        mask = torch.full_like(scores, float('-inf'))
        mask.scatter_(-1, idx, vals)
        probs = F.softmax(mask, dim=-1)
        return probs, idx

    def _hard_top1_capacity_route(self, logits, mask):
        B, N, K = logits.shape
        probs = F.softmax(logits / self.gate_temp, dim=-1)
        T = B * N
        logits_f = logits.reshape(T, K)
        probs_f = probs.reshape(T, K)
        if mask is not None:
            valid = mask.reshape(T)
        else:
            valid = torch.ones(T, dtype=torch.bool, device=logits.device)
        # Token chooses top-1 expert
        top1_idx = torch.argmax(logits_f, dim=-1)
        # Capacity per expert
        capacity = max(1, int(torch.ceil(self.capacity_factor * valid.sum() / K).item()))
        dispatch = torch.zeros(T, K, device=logits.device, dtype=torch.bool)
        load = torch.zeros(K, device=logits.device)
        for k in range(K):
            sel = (top1_idx == k) & valid
            if sel.any():
                scores_k = logits_f[sel, k]
                rank_idx = torch.argsort(scores_k, descending=True)
                # map back to original indices and keep top capacity
                token_ids = sel.nonzero(as_tuple=False).squeeze(1)
                keep = token_ids[rank_idx[:capacity]]
                dispatch[keep, k] = True
                load[k] = dispatch[:, k].sum() / capacity
            else:
                load[k] = 0.0
        importance = probs_f.mean(dim=0)  # soft importance
        dispatch = dispatch.reshape(B, N, K)
        return dispatch, importance, load

    def forward(self, h2d, h3d, mask=None, return_stats=False):
        # Inputs: [B,N,d2d], [B,N,d3d], mask [B,N] (True/1 for valid)
        x = torch.cat([h2d, h3d], dim=-1)  # [B,N,d2d+d3d]
        x = self.input_proj(x)             # [B,N,df]
        base = self.shared(x)              # [B,N,df]
        logits = self.gate_mlp(base)       # [B,N,K]
        logits = self._add_noise(logits)
        logits = self._mask_logits(logits, mask)
        z_loss = self._router_zloss(logits)

        stats = None
        if self.use_hard_capacity and self.topk == 1:
            # Switch-like routing with per-expert capacity
            dispatch, importance, load = self._hard_top1_capacity_route(logits, mask)
            out = torch.zeros_like(base)
            for k in range(self.K):
                m = dispatch[..., k]  # [B,N] bool
                if m.any():
                    y = self.expert_heads[k](base[m])
                    out[m] = y
            # Residual on base then project
            out = self.out_proj(out + base)
            if mask is not None:
                out = out * mask.unsqueeze(-1).float()
            if return_stats:
                stats = {
                    "importance": importance.detach(),
                    "load": load.detach(),
                    "z_loss": z_loss.detach(),
                }
            return out, stats
        else:
            # Top-k sparse weighting (fallback for k>1)
            probs, idx = self._topk_soft(logits)
            out = torch.zeros_like(base)
            for k in range(self.K):
                yk = self.expert_heads[k](base)
                p_k = probs[..., k].unsqueeze(-1)
                out = out + p_k * yk
            out = self.out_proj(out + base)
            if mask is not None:
                out = out * mask.unsqueeze(-1).float()
            if return_stats:
                importance = probs.mean(dim=(0, 1))
                entropy = (
                    -probs.clamp_min(1e-8) * probs.clamp_min(1e-8).log()
                ).sum(dim=-1).mean()
                stats = {
                    "importance": importance.detach(),
                    "entropy": entropy.detach(),
                    "z_loss": z_loss.detach(),
                }
            return out, stats
