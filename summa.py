"""
Parameter Golf — RTX 4060 (8GB) — RECURSIVE ATTENTION ARCHITECTURE
===================================================================
Novel Architecture: "Relaxed Recursive Transformer" (RRT)
Inspired by: Google DeepMind paper + parameter-golf PR #386

KEY IDEA: Instead of N independent layers (N × params),
use 1 SHARED block applied R times with per-step LoRA adapters.

  Standard 10-layer: 10 × 2.5M = 25M params (uses ~15MB compressed)
  Recursive 12-step:  1 × 2.5M + 12 × 0.2M = 4.9M params (uses ~3MB)
  → Leftover 13MB budget → bigger BigramHash, more recurrence steps!

Stack of proven leaderboard techniques on top:
  [#1] INT5 MLP + INT6 attn quantization + zstd-22
  [#1] BigramHash(10240) + SWA from 40%
  [#2] SmearGate + OrthoInit + Muon WD=0.04
  [#3] INT6 QAT (STE) + val_stride=64
  [NEW] Per-step LoRA adapters (rank=16) on Q,K,V,O,FC,PROJ
  [NEW] Learnable step embeddings (positional signal per recurrence step)
  [NEW] Adaptive halting gate (exit early on easy tokens)
  [NEW] Hidden state highway (direct skip from input to each step)

Run:
  PYTHONUTF8=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  RUN_ID=recursive_4060 python model.py
"""
from __future__ import annotations
import glob, io, math, os, platform, random, sys, time, uuid, zlib, warnings
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
from torch import Tensor, nn

# ── Windows / encoding fixes ──────────────────────────────────────────────────
if platform.system() == "Windows":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    warnings.filterwarnings("ignore", message="expandable_segments not supported")
    warnings.filterwarnings("ignore", category=FutureWarning)

# ── zstd compression ──────────────────────────────────────────────────────────
try:
    import zstandard as zstd
    def compress_bytes(b: bytes) -> bytes:
        return zstd.ZstdCompressor(level=22).compress(b)
    COMPRESSOR = "zstd-22"
except ImportError:
    def compress_bytes(b: bytes) -> bytes:
        return zlib.compress(b, level=9)
    COMPRESSOR = "zlib-9"

# ==============================================================================
# SECTION 1 — HYPERPARAMETERS
# ==============================================================================

class HP:
    # Paths
    data_path      = os.environ.get("DATA_PATH",      "./data/datasets/fineweb10B_sp1024")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_1024_bpe.model")
    run_id         = os.environ.get("RUN_ID",         str(uuid.uuid4()))
    seed           = int(os.environ.get("SEED",       "1337"))

    # Model
    vocab_size      = int(os.environ.get("VOCAB_SIZE",       "1024"))
    bigram_size     = int(os.environ.get("BIGRAM_SIZE",      "10240"))  # proven best
    model_dim       = int(os.environ.get("MODEL_DIM",        "512"))
    num_heads       = int(os.environ.get("NUM_HEADS",        "8"))
    num_kv_heads    = int(os.environ.get("NUM_KV_HEADS",     "4"))
    mlp_mult        = int(os.environ.get("MLP_MULT",         "3"))      # 3x MLP

    # ── RECURSIVE SETTINGS ──────────────────────────────────────────────────
    # num_recur_steps: how many times the shared block is applied
    num_recur_steps = int(os.environ.get("NUM_RECUR_STEPS",  "12"))     # 14 passes
    # Alias for logging / compatibility
    n_layer         = num_recur_steps
    # LoRA rank for per-step adapters
    lora_rank       = int(os.environ.get("LORA_RANK",        "16"))     # rank-16 adapters
    # Step embedding dim
    step_emb_dim    = int(os.environ.get("STEP_EMB_DIM",     "64"))
    # U-Net style encoder / decoder split
    num_enc_steps   = int(os.environ.get("NUM_ENC_STEPS",    "7"))      # 7 enc / 7 dec
    # Adaptive halting (inference only)
    use_halt        = bool(int(os.environ.get("USE_HALT",    "0")))
    halt_threshold  = float(os.environ.get("HALT_THRESHOLD", "0.9"))

    # Attention / positional
    rope_base       = float(os.environ.get("ROPE_BASE",      "500000"))
    logit_softcap   = float(os.environ.get("LOGIT_SOFTCAP",  "30.0"))
    qk_gain_init    = float(os.environ.get("QK_GAIN_INIT",   "1.5"))
    use_smeargate   = bool(int(os.environ.get("USE_SMEARGATE", "1")))

    # Training
    seq_len         = int(os.environ.get("TRAIN_SEQ_LEN",     "1024"))
    batch_tokens    = int(os.environ.get("TRAIN_BATCH_TOKENS","32768"))
    iterations      = int(os.environ.get("ITERATIONS",        "4000"))
    warmdown_iters  = int(os.environ.get("WARMDOWN_ITERS",    "600"))
    warmup_steps    = int(os.environ.get("WARMUP_STEPS",      "100"))
    val_every       = int(os.environ.get("VAL_LOSS_EVERY",    "400"))
    val_stride      = int(os.environ.get("VAL_STRIDE",        "64"))
    max_wallclock   = float(os.environ.get("MAX_WALLCLOCK_SECONDS", "0"))
    grad_ckpt       = bool(int(os.environ.get("GRAD_CKPT",    "1")))

    # Max sequences per batch (controls real batch size)
    max_seqs        = int(os.environ.get("MAX_SEQS",          "6"))

    # Checkpointing / patience
    ckpt_every      = int(os.environ.get("CKPT_EVERY",        "200"))  # steps
    patience_evals  = int(os.environ.get("PATIENCE_EVALS",    "5"))    # evals w/o improvement

    # SWA — start at 40%
    use_swa         = bool(int(os.environ.get("USE_SWA",      "1")))
    swa_start_frac  = float(os.environ.get("SWA_START_FRAC",  "0.4"))
    swa_every       = int(os.environ.get("SWA_EVERY",         "20"))

    # QAT
    use_qat         = bool(int(os.environ.get("USE_QAT",      "1")))
    qat_start_frac  = float(os.environ.get("QAT_START_FRAC",  "0.1"))
    qat_bits_attn   = int(os.environ.get("QAT_BITS_ATTN",     "6"))
    qat_bits_mlp    = int(os.environ.get("QAT_BITS_MLP",      "5"))

    # Optimizer
    matrix_lr       = float(os.environ.get("MATRIX_LR",   "0.04"))
    lora_lr         = float(os.environ.get("LORA_LR",     "0.06"))  # LoRA trains faster
    embed_lr        = float(os.environ.get("EMBED_LR",    "0.05"))
    scalar_lr       = float(os.environ.get("SCALAR_LR",   "0.04"))
    muon_wd         = float(os.environ.get("MUON_WD",     "0.04"))
    beta1           = float(os.environ.get("BETA1",       "0.9"))
    beta2           = float(os.environ.get("BETA2",       "0.95"))
    embed_init_std  = float(os.environ.get("EMBED_INIT_STD", "0.005"))

# ==============================================================================
# SECTION 2 — MUON OPTIMIZER (Nesterov + decoupled WD)
# ==============================================================================

def newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.bfloat16() / (G.norm() + eps)
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        X = a * X + (b * A + c * A @ A) @ X
    return X.T if transposed else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.95, wd=0.04, steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, wd=wd, steps=steps))
        for g in self.param_groups:
            g["base_lr"] = lr

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            lr, mom, wd, steps = (
                group["lr"], group["momentum"], group["wd"], group["steps"]
            )
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(g)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom)  # Nesterov
                g = newtonschulz5(g, steps=steps)
                g *= max(1.0, g.size(0) / g.size(1)) ** 0.5
                p.add_(g, alpha=-lr)
                if wd > 0:
                    p.mul_(1.0 - lr * wd)

# ==============================================================================
# SECTION 3 — QAT (STE)
# ==============================================================================

class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, bits: int) -> Tensor:
        levels = (1 << (bits - 1)) - 1
        scale = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / levels
        return (x / scale).round_().clamp_(-levels, levels) * scale

    @staticmethod
    def backward(ctx, grad):
        return grad, None

def qat_q(x: Tensor, bits: int, active: bool) -> Tensor:
    return STE.apply(x, bits) if active else x

# ==============================================================================
# SECTION 4 — INT5/INT6 PACKING
# ==============================================================================

def pack_int6(x: Tensor) -> Tensor:
    flat = x.reshape(-1)
    pad = (-flat.numel()) % 4
    if pad:
        flat = F.pad(flat.float(), (0, pad)).to(torch.int8)
    u = (flat.to(torch.int32) + 32).to(torch.uint8).reshape(-1, 4)
    v = [u[:, i] for i in range(4)]
    b0 = ((v[0] << 2) | (v[1] >> 4)).to(torch.uint8)
    b1 = (((v[1] & 0xF) << 4) | (v[2] >> 2)).to(torch.uint8)
    b2 = (((v[2] & 0x3) << 6) | v[3]).to(torch.uint8)
    return torch.stack([b0, b1, b2], dim=1).reshape(-1)

def unpack_int6(pk: Tensor, n: int) -> Tensor:
    p = pk.reshape(-1, 3)
    b0, b1, b2 = p[:, 0].int(), p[:, 1].int(), p[:, 2].int()
    v0 = b0 >> 2
    v1 = ((b0 & 3) << 4) | (b1 >> 4)
    v2 = ((b1 & 0xF) << 2) | (b2 >> 6)
    v3 = b2 & 0x3F
    return (torch.stack([v0, v1, v2, v3], 1).reshape(-1)[:n] - 32).to(torch.int8)

def pack_int5(x: Tensor) -> Tensor:
    flat = x.reshape(-1)
    pad = (-flat.numel()) % 8
    if pad:
        flat = F.pad(flat.float(), (0, pad)).to(torch.int8)
    u = (flat.to(torch.int32) + 16).to(torch.uint8).reshape(-1, 8)
    b0 = ((u[:, 0] << 3) | (u[:, 1] >> 2)).to(torch.uint8)
    b1 = (((u[:, 1] & 3) << 6) | (u[:, 2] << 1) | (u[:, 3] >> 4)).to(torch.uint8)
    b2 = (((u[:, 3] & 0xF) << 4) | (u[:, 4] >> 1)).to(torch.uint8)
    b3 = (((u[:, 4] & 1) << 7) | (u[:, 5] << 2) | (u[:, 6] >> 3)).to(torch.uint8)
    b4 = (((u[:, 6] & 7) << 5) | u[:, 7]).to(torch.uint8)
    return torch.stack([b0, b1, b2, b3, b4], 1).reshape(-1)

def unpack_int5(pk: Tensor, n: int) -> Tensor:
    p = pk.reshape(-1, 5)
    b = [p[:, i].int() for i in range(5)]
    v0 = b[0] >> 3
    v1 = ((b[0] & 7) << 2) | (b[1] >> 6)
    v2 = (b[1] >> 1) & 31
    v3 = ((b[1] & 1) << 4) | (b[2] >> 4)
    v4 = ((b[2] & 0xF) << 1) | (b[3] >> 7)
    v5 = (b[3] >> 2) & 31
    v6 = ((b[3] & 3) << 3) | (b[4] >> 5)
    v7 = b[4] & 31
    return (torch.stack([v0, v1, v2, v3, v4, v5, v6, v7], 1).reshape(-1)[:n] - 16).to(torch.int8)

CTRL_KEYS = ("mix", "as_", "ms_", "q_gain", "skip_w", "halt", "step_emb", "bigr", "scale")

def quantize_state_dict(sd: dict):
    out, meta = {}, {}
    for name, t in sd.items():
        t = t.detach().cpu().float()
        small = t.numel() <= 4096
        is_ctrl = any(k in name for k in CTRL_KEYS)
        is_lora = "lora" in name
        if not t.is_floating_point() or t.ndim != 2 or small or is_ctrl or is_lora:
            out[name] = t.half().contiguous()
            meta[name] = "fp16"
            continue
        is_mlp = ("fc." in name or ".proj." in name) and "mlp" in name
        r, c = t.shape
        if is_mlp:  # INT5 for MLP
            clip = torch.quantile(t.abs(), 0.99995, dim=1)
            tc = t.clamp(-clip[:, None], clip[:, None])
            sc = (clip / 15.0).clamp_min(1 / 15.0)
            q = tc.div(sc[:, None]).round_().clamp(-15, 15).to(torch.int8)
            out[name] = pack_int5(q)
            meta[name] = {"s": "int5", "sh": [r, c], "sc": sc.half()}
        else:       # INT6 for attention
            clip = torch.quantile(t.abs(), 0.99998, dim=1)
            tc = t.clamp(-clip[:, None], clip[:, None])
            sc = (clip / 31.0).clamp_min(1 / 31.0)
            q = tc.div(sc[:, None]).round_().clamp(-31, 31).to(torch.int8)
            out[name] = pack_int6(q)
            meta[name] = {"s": "int6", "sh": [r, c], "sc": sc.half()}
    return out, meta

def dequantize_state_dict(qsd, meta):
    out = {}
    for name, t in qsd.items():
        m = meta[name]
        if m == "fp16":
            out[name] = t.float()
            continue
        sc = m["sc"].float()
        sh = m["sh"]
        n = sh[0] * sh[1]
        q = (unpack_int5(t, n) if m["s"] == "int5" else unpack_int6(t, n)).reshape(sh)
        out[name] = (q.float() * sc[:, None]).contiguous()
    return out

# ==============================================================================
# SECTION 5 — BIGRAM HASH, SMEARGATE, LORA, STEP EMB, HALT GATE
# ==============================================================================

class BigramEmb(nn.Module):
    def __init__(self, vocab: int, hash_size: int, dim: int):
        super().__init__()
        self.h = hash_size
        self.emb = nn.Embedding(hash_size, dim)
        nn.init.normal_(self.emb.weight, std=0.002)

    def forward(self, x: Tensor) -> Tensor:
        prev = torch.cat([x[:, :1], x[:, :-1]], dim=1)
        return self.emb((prev.long() * 1_000_003 + x.long()) % self.h)

class SmearGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, dim, bias=False)
        nn.init.zeros_(self.gate.weight)

    def forward(self, x: Tensor) -> Tensor:
        g = torch.sigmoid(self.gate(x.float()).to(x.dtype))
        prev = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        return x * (1 - g) + prev * g

class LoRAAdapter(nn.Module):
    """Single LoRA adapter: ΔW = B @ A, initialized so ΔW = 0."""
    def __init__(self, in_dim: int, out_dim: int, rank: int):
        super().__init__()
        self.A = nn.Parameter(torch.empty(rank, in_dim))
        self.B = nn.Parameter(torch.zeros(out_dim, rank))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(F.linear(x, self.A), self.B)

class StepLoRABank(nn.Module):
    """
    Bank of LoRA adapters for all weight matrices × all recurrence steps.
    Covers: Q, K, V, O (attention) + FC, PROJ (MLP) = 6 matrices.
    """
    def __init__(self, dim: int, mlp_hidden: int, num_steps: int, rank: int):
        super().__init__()
        self.num_steps = num_steps
        def mk(in_d, out_d):
            return nn.ModuleList([LoRAAdapter(in_d, out_d, rank) for _ in range(num_steps)])
        self.q_loras    = mk(dim, dim)
        self.k_loras    = mk(dim, dim // 2)  # GQA KV heads = NH/2
        self.v_loras    = mk(dim, dim // 2)
        self.o_loras    = mk(dim, dim)
        self.fc_loras   = mk(dim, mlp_hidden)
        self.proj_loras = mk(mlp_hidden, dim)

    def get(self, step: int):
        return {
            "q": self.q_loras[step], "k": self.k_loras[step],
            "v": self.v_loras[step], "o": self.o_loras[step],
            "fc": self.fc_loras[step], "proj": self.proj_loras[step],
        }

class StepEmbedding(nn.Module):
    def __init__(self, num_steps: int, dim: int, step_emb_dim: int):
        super().__init__()
        self.emb  = nn.Embedding(num_steps, step_emb_dim)
        self.proj = nn.Linear(step_emb_dim, dim, bias=False)
        nn.init.normal_(self.emb.weight, std=0.01)
        nn.init.zeros_(self.proj.weight)

    def forward(self, step: int, B: int, T: int, device) -> Tensor:
        idx = torch.tensor(step, device=device)
        se  = self.proj(self.emb(idx))
        return se[None, None, :].expand(B, T, -1)

class HaltGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 1, bias=True)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -3.0)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sigmoid(self.gate(x.float()).to(x.dtype))

# ==============================================================================
# SECTION 10 — SHARED BLOCK
# ==============================================================================

class CastedLinear(nn.Linear):
    def __init__(self, *a, zero_init=False, **kw):
        super().__init__(*a, **kw)
        self._zero_init = zero_init

    def forward(self, x):
        b = self.bias.to(x.dtype) if self.bias is not None else None
        return F.linear(x, self.weight.to(x.dtype), b)

class RMSNorm(nn.Module):
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),))

class Rotary(nn.Module):
    def __init__(self, dim, base):
        super().__init__()
        self.register_buffer(
            "inv_freq",
            1 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)),
            persistent=False,
        )
        self._T = 0
        self._cos = self._sin = None

    def forward(self, T, device, dtype):
        if self._T != T or self._cos is None or self._cos.device != device:
            t = torch.arange(T, device=device, dtype=self.inv_freq.dtype)
            f = torch.outer(t, self.inv_freq.to(device))
            self._cos = f.cos()[None, None]
            self._sin = f.sin()[None, None]
            self._T = T
        return self._cos.to(dtype), self._sin.to(dtype)

def rope_rotate(x, cos, sin):
    h = x.size(-1) // 2
    return torch.cat(
        (x[..., :h] * cos - x[..., h:] * sin,
         x[..., :h] * sin + x[..., h:] * cos),
        dim=-1,
    )

class SharedAttention(nn.Module):
    def __init__(self, dim, nh, nkv, rope_base, qk_init):
        super().__init__()
        self.nh = nh
        self.nkv = nkv
        self.hd = dim // nh
        self.c_q = CastedLinear(dim, dim, bias=False)
        self.c_k = CastedLinear(dim, nkv * self.hd, bias=False)
        self.c_v = CastedLinear(dim, nkv * self.hd, bias=False)
        self.proj = CastedLinear(dim, dim, bias=False, zero_init=True)
        self.q_gain = nn.Parameter(torch.full((nh,), qk_init))
        self.rotary = Rotary(self.hd, rope_base)
        self.qat_bits = 0

    def forward(self, x, loras: dict, qat_active: bool) -> Tensor:
        B, T, D = x.shape
        Wq = self.c_q.weight.to(x.dtype)
        Wk = self.c_k.weight.to(x.dtype)
        Wv = self.c_v.weight.to(x.dtype)
        Wo = self.proj.weight.to(x.dtype)
        if qat_active and self.qat_bits > 0:
            Wq = qat_q(Wq, self.qat_bits, True)
            Wk = qat_q(Wk, self.qat_bits, True)
            Wv = qat_q(Wv, self.qat_bits, True)
            Wo = qat_q(Wo, self.qat_bits, True)

        q = F.linear(x, Wq) + loras["q"](x)
        k = F.linear(x, Wk) + loras["k"](x)
        v = F.linear(x, Wv) + loras["v"](x)

        q = q.reshape(B, T, self.nh, self.hd).transpose(1, 2)
        k = k.reshape(B, T, self.nkv, self.hd).transpose(1, 2)
        v = v.reshape(B, T, self.nkv, self.hd).transpose(1, 2)

        q = F.rms_norm(q, (q.size(-1),))
        k = F.rms_norm(k, (k.size(-1),))
        cos, sin = self.rotary(T, x.device, q.dtype)
        q = rope_rotate(q, cos, sin)
        k = rope_rotate(k, cos, sin)
        q = q * self.q_gain.to(q.dtype)[None, :, None, None]

        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = y.transpose(1, 2).contiguous().reshape(B, T, D)
        return F.linear(o, Wo) + loras["o"](o)

class SharedMLP(nn.Module):
    def __init__(self, dim, mult):
        super().__init__()
        self.h = mult * dim
        self.fc   = CastedLinear(dim, self.h, bias=False)
        self.proj = CastedLinear(self.h, dim, bias=False, zero_init=True)
        self.qat_bits = 0

    def forward(self, x, loras: dict, qat_active: bool) -> Tensor:
        Wfc   = self.fc.weight.to(x.dtype)
        Wproj = self.proj.weight.to(x.dtype)
        if qat_active and self.qat_bits > 0:
            Wfc   = qat_q(Wfc,   self.qat_bits, True)
            Wproj = qat_q(Wproj, self.qat_bits, True)
        h = F.linear(x, Wfc) + loras["fc"](x)
        h = torch.relu(h).square()
        return F.linear(h, Wproj) + loras["proj"](h)

class SharedBlock(nn.Module):
    """One transformer block. Applied R times via RecursiveGPT."""
    def __init__(self, dim, nh, nkv, mult, rope_base, qk_init, use_smear):
        super().__init__()
        self.an   = RMSNorm()
        self.mn   = RMSNorm()
        self.attn = SharedAttention(dim, nh, nkv, rope_base, qk_init)
        self.mlp  = SharedMLP(dim, mult)
        self.smear = SmearGate(dim) if use_smear else None
        self.as_  = nn.Parameter(torch.ones(dim))
        self.ms_  = nn.Parameter(torch.ones(dim))
        self.mix  = nn.Parameter(torch.stack([torch.ones(dim), torch.zeros(dim)]).float())

    def set_qat(self, enabled, bits_attn, bits_mlp):
        self.attn.qat_bits = bits_attn if enabled else 0
        self.mlp.qat_bits  = bits_mlp  if enabled else 0

    def forward(self, x: Tensor, x0: Tensor, loras: dict,
                step_delta: Tensor, qat_active: bool) -> Tensor:
        m = self.mix.to(x.dtype)
        x = m[0][None, None, :] * x + m[1][None, None, :] * x0
        x = x + step_delta
        ao = self.attn(self.an(x), loras, qat_active)
        if self.smear:
            ao = self.smear(ao)
        x = x + self.as_.to(x.dtype)[None, None, :] * ao
        x = x + self.ms_.to(x.dtype)[None, None, :] * self.mlp(self.mn(x), loras, qat_active)
        return x

# ==============================================================================
# SECTION 11 — RECURSIVE GPT MODEL
# ==============================================================================

class RecursiveGPT(nn.Module):
    def __init__(self, hp: HP):
        super().__init__()
        D = hp.model_dim
        MH = hp.mlp_mult * D

        self.cap       = hp.logit_softcap
        self.ne        = hp.num_enc_steps
        self.nd        = hp.num_recur_steps - hp.num_enc_steps
        self.n_skip    = min(self.ne, self.nd)
        self.n_steps   = hp.num_recur_steps
        self.use_halt  = hp.use_halt
        self.halt_thr  = hp.halt_threshold
        self.qat_active = False

        self.emb  = nn.Embedding(hp.vocab_size, D)
        self.bigr = BigramEmb(hp.vocab_size, hp.bigram_size, D)

        self.block = SharedBlock(D, hp.num_heads, hp.num_kv_heads,
                                 hp.mlp_mult, hp.rope_base,
                                 hp.qk_gain_init, hp.use_smeargate)

        self.lora_bank = StepLoRABank(D, MH, hp.num_recur_steps, hp.lora_rank)
        self.step_emb  = StepEmbedding(hp.num_recur_steps, D, hp.step_emb_dim)
        self.skip_w    = nn.Parameter(torch.ones(self.n_skip, D))

        if hp.use_halt:
            self.halt = HaltGate(D)

        self.norm      = RMSNorm()
        self.grad_ckpt = hp.grad_ckpt
        self._init(hp)

    def _init(self, hp: HP):
        V, D = hp.vocab_size, hp.model_dim
        Q, _ = torch.linalg.qr(torch.randn(D, D))
        W = torch.zeros(V, D)
        W[:D] = Q * hp.embed_init_std * (D ** 0.5)
        W[D:] = torch.randn(V - D, D) * hp.embed_init_std
        with torch.no_grad():
            self.emb.weight.copy_(W)

        for m in self.modules():
            if not isinstance(m, CastedLinear):
                continue
            r, c = m.weight.shape
            if m._zero_init:
                nn.init.zeros_(m.weight)
                continue
            if r >= c:
                Q2, _ = torch.linalg.qr(torch.randn(r, c))
            else:
                Qt, _ = torch.linalg.qr(torch.randn(c, r))
                Q2 = Qt.T
            with torch.no_grad():
                m.weight.copy_(Q2 * math.sqrt(2.0 / (r + c)))

    def set_qat(self, enabled: bool, hp: HP):
        self.qat_active = enabled
        self.block.set_qat(enabled, hp.qat_bits_attn, hp.qat_bits_mlp)

    def _step(self, x: Tensor, x0: Tensor, step: int) -> Tensor:
        loras      = self.lora_bank.get(step)
        step_delta = self.step_emb(step, x.size(0), x.size(1), x.device)
        return self.block(x, x0, loras, step_delta, self.qat_active)

    def forward(self, ids: Tensor, tgt: Tensor,
                return_per_token: bool = False) -> Tensor:
        B, T = ids.shape
        x  = F.rms_norm(self.emb(ids), (self.emb.embedding_dim,))
        x  = x + self.bigr(ids)
        x0 = x

        skips: list[Tensor] = []
        halt_acc = None

        # Encoder steps
        for s in range(self.ne):
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._step, x, x0, s, use_reentrant=False
                )
            else:
                x = self._step(x, x0, s)
            skips.append(x.clone())

        # Decoder steps
        for s in range(self.nd):
            si = min(s, self.n_skip - 1)
            x  = x + self.skip_w[si].to(x.dtype)[None, None, :] * skips.pop()

            step_idx = self.ne + s
            if self.grad_ckpt and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    self._step, x, x0, step_idx, use_reentrant=False
                )
            else:
                x = self._step(x, x0, step_idx)

            if self.use_halt and not self.training:
                h = self.halt(x)
                if halt_acc is None:
                    halt_acc = h
                else:
                    halt_acc = halt_acc + (1 - halt_acc) * h
                if halt_acc.mean().item() > self.halt_thr:
                    break

        x = self.norm(x)
        logits = self.cap * torch.tanh(F.linear(x, self.emb.weight) / self.cap)

        B, T, V = logits.shape
        if return_per_token:
            return F.cross_entropy(
                logits.float().reshape(-1, V), tgt.reshape(-1),
                reduction="none"
            ).reshape(B, T)
        return F.cross_entropy(logits.float().reshape(-1, V), tgt.reshape(-1))

# ==============================================================================
# SECTION 12 — SWA
# ==============================================================================

class SWA:
    def __init__(self, model):
        self.n   = 0
        self.avg = {
            k: v.clone().float()
            for k, v in model.state_dict().items()
            if v.is_floating_point()
        }

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        for k, v in model.state_dict().items():
            if k in self.avg:
                self.avg[k].add_(v.float() - self.avg[k], alpha=1.0 / self.n)

    def apply(self, model):
        sd = model.state_dict()
        for k, v in self.avg.items():
            if k in sd:
                sd[k].copy_(v.to(sd[k].dtype))
        model.load_state_dict(sd)

# ==============================================================================
# SECTION 13 — DATA + EVAL
# ==============================================================================

def load_shard(p: Path) -> Tensor:
    h = np.fromfile(p, dtype=np.int32, count=256)
    assert h[0] == 20240520
    return torch.from_numpy(
        np.fromfile(p, dtype=np.uint16, count=int(h[2]), offset=256 * 4)
        .astype(np.int32)
    )

class DataLoader:
    def __init__(self, pattern, device):
        self.files = [Path(p) for p in sorted(glob.glob(pattern))]
        assert self.files, f"No files: {pattern}"
        self.device = device
        self.fi = 0
        self.pos = 0
        self.tok = load_shard(self.files[0])

    def _adv(self):
        self.fi = (self.fi + 1) % len(self.files)
        self.tok = load_shard(self.files[self.fi])
        self.pos = 0

    def batch(self, n_tok, seq_len, max_seqs=6):
        needed = n_tok + 1
        chunks = []
        while needed > 0:
            av = self.tok.numel() - self.pos
            if av <= 0:
                self._adv()
                continue
            k = min(needed, av)
            chunks.append(self.tok[self.pos:self.pos + k])
            self.pos += k
            needed -= k
        seg = (chunks[0] if len(chunks) == 1 else torch.cat(chunks)).to(torch.int64)
        x = seg[:-1].reshape(-1, seq_len)[:max_seqs]
        y = seg[1:].reshape(-1, seq_len)[:max_seqs]
        return (
            x.to(self.device, non_blocking=True),
            y.to(self.device, non_blocking=True),
        )

def build_luts(sp, vocab_size, device):
    sz = max(int(sp.vocab_size()), vocab_size)
    bb = np.zeros(sz, dtype=np.int16)
    hs = np.zeros(sz, dtype=np.bool_)
    ib = np.ones(sz, dtype=np.bool_)
    for i in range(int(sp.vocab_size())):
        if sp.is_control(i) or sp.is_unknown(i) or sp.is_unused(i):
            continue
        ib[i] = False
        if sp.is_byte(i):
            bb[i] = 1
            continue
        p = sp.id_to_piece(i)
        if p.startswith("▁"):
            hs[i] = True
            p = p[1:]
        bb[i] = len(p.encode("utf-8"))
    return (
        torch.tensor(bb, dtype=torch.int16, device=device),
        torch.tensor(hs, dtype=torch.bool, device=device),
        torch.tensor(ib, dtype=torch.bool, device=device),
    )

@torch.no_grad()
def eval_bpb(model, val_tok, bb, hs, ib, hp, device):
    model.eval()
    W, S = hp.seq_len, hp.val_stride
    N = val_tok.numel() - 1
    starts = list(range(0, N - W + 1, S))
    bsz = max(1, 4096 // W)
    ls = ts = bs_ = 0.0
    for bi in range(0, len(starts), bsz):
        sl = starts[bi:bi + bsz]
        xs = torch.stack([val_tok[s:s + W].to(device, torch.int64) for s in sl])
        ys = torch.stack([val_tok[s + 1:s + W + 1].to(device, torch.int64) for s in sl])
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pt = model(xs, ys, return_per_token=True)
        for i, s in enumerate(sl):
            off = 0 if s == 0 else (W - S)
            tl = pt[i, off:].float()
            ti = ys[i, off:]
            pi = xs[i, off:]
            ls += tl.sum().item()
            ts += tl.numel()
            bs_ += (bb[ti].float() + (hs[ti] & ~ib[pi]).float()).sum().item()
    model.train()
    loss = ls / ts
    bpb = loss / math.log(2) * (ts / bs_)
    return loss, bpb

def get_lr_scale(step, total, warmdown, warmup):
    if step < warmup:
        return step / max(warmup, 1)
    wd_start = total - warmdown
    if step >= wd_start:
        return max((total - step) / max(warmdown, 1), 0.0)
    return 1.0

# ==============================================================================
# SECTION 14 — MAIN (with checkpoints & patience)
# ==============================================================================

def main():
    hp = HP()

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import (
        enable_flash_sdp, enable_math_sdp,
        enable_mem_efficient_sdp, enable_cudnn_sdp
    )
    # SDP choices kept conservative for Windows
    enable_flash_sdp(False)
    enable_cudnn_sdp(False)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu = torch.cuda.get_device_name(device) if torch.cuda.is_available() else "CPU"

    random.seed(hp.seed)
    np.random.seed(hp.seed)
    torch.manual_seed(hp.seed)
    torch.cuda.manual_seed_all(hp.seed)

    os.makedirs("logs", exist_ok=True)
    logfile = f"logs/{hp.run_id}.txt"

    def log(msg):
        print(msg)
        with open(logfile, "a", encoding="utf-8") as f:
            print(msg, file=f)

    log(f"Device : {gpu}")
    log(f"Compress: {COMPRESSOR}")

    sp = spm.SentencePieceProcessor(model_file=hp.tokenizer_path)
    val_files = sorted(glob.glob(os.path.join(hp.data_path, "fineweb_val_*.bin")))
    assert val_files
    val_tok = torch.cat([load_shard(Path(f)) for f in val_files])
    usable = ((val_tok.numel() - 1) // hp.seq_len) * hp.seq_len
    val_tok = val_tok[:usable + 1]
    bb, hs, ib = build_luts(sp, hp.vocab_size, device)

    # Build model
    model = RecursiveGPT(hp).to(device).bfloat16()

    # Keep scalars + control params in fp32
    ctrl = ("mix", "as_", "ms_", "q_gain", "skip_w", "halt", "step_emb", "scale")
    for name, p in model.named_parameters():
        if p.ndim < 2 or any(k in name for k in ctrl):
            p.data = p.data.float()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()

    # Parameter counts
    base_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "lora_bank" not in n and "step_emb" not in n
    )
    lora_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "lora_bank" in n
    )
    step_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "step_emb" in n
    )
    total_params = sum(p.numel() for p in model.parameters())

    log(f"\nArchitecture: Recursive × {hp.num_recur_steps} steps "
        f"(layers={hp.n_layer})")
    log(f"  Base block:   {base_params/1e6:.2f}M params (shared)")
    log(f"  LoRA adapters:{lora_params/1e6:.2f}M params "
        f"(rank={hp.lora_rank} × {hp.num_recur_steps} steps × 6 matrices)")
    log(f"  Step embeddings:{step_params/1e3:.0f}K params")
    log(f"  TOTAL:        {total_params/1e6:.2f}M params")
    log(f"  Config: dim={hp.model_dim} mlp={hp.mlp_mult}x "
        f"seq={hp.seq_len} bigram={hp.bigram_size}")
    log(f"  QAT: INT{hp.qat_bits_attn}/INT{hp.qat_bits_mlp} "
        f"| SWA@{hp.swa_start_frac:.0%} | smear={hp.use_smeargate}")

    # Optimizers
    ctrl_set = set(ctrl)
    base_mats = [
        p for n, p in model.block.named_parameters()
        if p.ndim == 2 and not any(k in n for k in ctrl_set)
    ]
    lora_mats = [
        p for n, p in model.lora_bank.named_parameters()
        if p.ndim == 2 and "A" in n
    ]
    lora_b = [
        p for n, p in model.lora_bank.named_parameters()
        if p.ndim == 2 and "B" in n
    ]
    scals = [
        p for n, p in model.named_parameters()
        if p.ndim < 2 or any(k in n for k in ctrl_set)
    ] + lora_b

    opt_muon = Muon(base_mats + lora_mats, lr=hp.matrix_lr, wd=hp.muon_wd)
    opt_emb = torch.optim.Adam(
        list(model.emb.parameters()) + list(model.bigr.parameters())
        + list(model.step_emb.parameters()),
        lr=hp.embed_lr, betas=(hp.beta1, hp.beta2),
        fused=torch.cuda.is_available(),
    )
    opt_scal = torch.optim.Adam(
        scals, lr=hp.scalar_lr, betas=(hp.beta1, hp.beta2),
        fused=torch.cuda.is_available(),
    )
    optimizers = [opt_muon, opt_emb, opt_scal]
    for opt in optimizers:
        for g in opt.param_groups:
            g["base_lr"] = g["lr"]

    swa       = SWA(model) if hp.use_swa else None
    swa_start = int(hp.iterations * hp.swa_start_frac)
    qat_start = int(hp.iterations * hp.qat_start_frac) if hp.use_qat else hp.iterations + 1
    qat_on    = False

    train_loader = DataLoader(
        os.path.join(hp.data_path, "fineweb_train_*.bin"), device
    )
    max_ms = hp.max_wallclock * 1000 if hp.max_wallclock > 0 else None
    t0     = time.perf_counter()

    # ── Checkpoint dir & helpers ─────────────────────────────────────────────
    ckpt_dir = Path("checkpoints") / hp.run_id
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    def _ckpt_state(step, best_val_bpb):
        return {
            "step": step,
            "model": model.state_dict(),
            "optimizers": [opt.state_dict() for opt in optimizers],
            "hp": hp.__dict__,
            "best_val_bpb": best_val_bpb,
            "swa_n": swa.n if swa is not None else 0,
            "swa_avg": swa.avg if swa is not None else None,
        }

    def save_latest(step, best_val_bpb):
        torch.save(_ckpt_state(step, best_val_bpb), ckpt_dir / "latest.pt")

    def save_best(step, best_val_bpb):
        torch.save(
            _ckpt_state(step, best_val_bpb),
            ckpt_dir / f"best_step_{step:06d}_bpb_{best_val_bpb:.4f}.pt",
        )

    # ── Resume (optional) ────────────────────────────────────────────────────
    resume_path = os.environ.get("RESUME_CKPT", "")
    start_step = 1
    best_val_bpb = float("inf")
    best_step = 0
    no_improve_evals = 0

    if resume_path:
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizers" in ckpt:
            for opt, state in zip(optimizers, ckpt["optimizers"]):
                opt.load_state_dict(state)
        start_step = ckpt.get("step", 0) + 1
        best_val_bpb = ckpt.get("best_val_bpb", best_val_bpb)
        if swa is not None and ckpt.get("swa_avg") is not None:
            swa.avg = {k: v.clone().float() for k, v in ckpt["swa_avg"].items()}
            swa.n   = ckpt.get("swa_n", swa.n)
        log(f"Resumed from {resume_path} at step {start_step - 1}")

    log(f"\nTraining {hp.iterations} steps | "
        f"QAT@step {qat_start} | SWA@step {swa_start}\n")

    step = 0
    try:
        for step in range(start_step, hp.iterations + 1):
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if max_ms and elapsed_ms > max_ms:
                log(f"Wallclock cap at step {step}")
                break

            # Enable QAT
            if hp.use_qat and step == qat_start:
                model.set_qat(True, hp)
                qat_on = True
                log(
                    f"step:{step} >> QAT ON (INT{hp.qat_bits_attn} attn / "
                    f"INT{hp.qat_bits_mlp} MLP)"
                )

            # LR schedule
            scale = get_lr_scale(step, hp.iterations, hp.warmdown_iters, hp.warmup_steps)
            for opt in optimizers:
                for g in opt.param_groups:
                    g["lr"] = g["base_lr"] * scale

            # Forward + backward
            for opt in optimizers:
                opt.zero_grad(set_to_none=True)
            x, y = train_loader.batch(hp.batch_tokens, hp.seq_len, max_seqs=hp.max_seqs)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            for opt in optimizers:
                opt.step()

            # SWA
            if swa and step >= swa_start and step % hp.swa_every == 0:
                swa.update(model)

            # Periodic logging
            if step % 100 == 0 or step == start_step:
                mem = torch.cuda.memory_allocated(device) / 1e9
                log(
                    f"step:{step:5d}/{hp.iterations} loss:{loss.item():.4f} "
                    f"lr:{scale:.3f} mem:{mem:.2f}GB qat:{qat_on} "
                    f"t:{elapsed_ms/1000:.0f}s"
                )

            # Periodic checkpoint
            if step % hp.ckpt_every == 0:
                save_latest(step, best_val_bpb)
                log(f"Checkpoint saved at step {step}")

            # Validation + best checkpoint + patience
            if hp.val_every > 0 and step % hp.val_every == 0:
                if swa and swa.n > 0:
                    backup = {k: v.clone() for k, v in model.state_dict().items()}
                    swa.apply(model)
                vl, vbpb = eval_bpb(model, val_tok, bb, hs, ib, hp, device)
                log(
                    f"  >> val_loss:{vl:.4f} val_bpb:{vbpb:.4f}"
                    f"{'  [SWA]' if swa and swa.n > 0 else ''}"
                )
                if swa and swa.n > 0:
                    model.load_state_dict(backup)

                if vbpb < best_val_bpb:
                    best_val_bpb = vbpb
                    best_step = step
                    no_improve_evals = 0
                    save_best(step, best_val_bpb)
                    log(
                        f"  >> New BEST bpb={vbpb:.4f} at step {step}, "
                        f"checkpoint saved"
                    )
                else:
                    no_improve_evals += 1
                    log(
                        f"  >> No improvement for "
                        f"{no_improve_evals}/{hp.patience_evals} evals"
                    )
                    if no_improve_evals >= hp.patience_evals:
                        log(
                            f"Early stopping: no val_bpb improvement for "
                            f"{hp.patience_evals} evals "
                            f"(best {best_val_bpb:.4f} at step {best_step})"
                        )
                        break

    except KeyboardInterrupt:
        log("Interrupted by user, saving emergency checkpoint...")
        save_latest(step, best_val_bpb)
        log(f"Emergency checkpoint saved at step {step} to "
            f"{(Path('checkpoints')/hp.run_id/'latest.pt').as_posix()}")

    # Final SWA + eval
    if swa and swa.n > 0:
        swa.apply(model)
        log(f"\nSWA applied ({swa.n} snapshots averaged)")
    vl, vbpb = eval_bpb(model, val_tok, bb, hs, ib, hp, device)
    log("\n" + "=" * 60)
    log(f"FINAL  val_loss:{vl:.4f}  val_bpb:{vbpb:.4f}")

    # Artifact size
    qsd, meta = quantize_state_dict(model.state_dict())
    buf = io.BytesIO()
    torch.save({"qsd": qsd, "meta": meta}, buf)
    compressed = compress_bytes(buf.getvalue())
    code_bytes = len(Path(__file__).read_bytes())
    total = code_bytes + len(compressed)
    log(
        f"\nArtifact: {total/1e6:.3f} MB / 16.000 MB "
        f"(headroom={max(0, 16e6 - total)/1e3:.0f}KB)"
    )
    log(
        f"  code={code_bytes/1e3:.0f}KB | "
        f"weights_compressed={len(compressed)/1e6:.2f}MB | "
        f"{COMPRESSOR}"
    )
    if total > 16_000_000:
        log("WARNING: Over budget! Reduce LORA_RANK or BIGRAM_SIZE")
    log("=" * 60)

if __name__ == "__main__":
    main()
