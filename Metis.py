import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np
from einops import rearrange

##----------------------------------------------------------------------------------------------------------------------
##----------------------------------------------MoE---------------------------------------------------------------------
class Gate(nn.Module):
    def __init__(self, dim, topk=2, n_routed_experts=8):
        super().__init__()
        self.dim = dim
        self.topk = topk
        self.n_routed_experts = n_routed_experts
        self.score_func = "softmax"
        self.route_scale = 1
        self.weight = nn.Parameter(torch.empty(n_routed_experts, dim))
        self.bias = nn.Parameter(torch.empty( n_routed_experts))
        self.alpha = 0.0001

    def forward(self, x: torch.Tensor):
        scores = F.linear(x, self.weight)
        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            scores = scores.sigmoid()
        original_scores = scores
        scores = scores + self.bias
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        weights = original_scores.gather(1, indices)
        if self.score_func == "sigmoid":
            weights /= weights.sum(dim=-1, keepdim=True)
        weights *= self.route_scale
        # 计算辅助损失（如果 alpha > 0 且是训练阶段）
        if self.training and self.alpha > 0:
            mask_ce = F.one_hot(indices.view(-1), num_classes=self.n_routed_experts)
            ce = mask_ce.float().mean(0)  # 每个专家被选中的概率
            Pi = original_scores.mean(0)
            fi = ce * self.n_routed_experts
            aux_loss = (Pi * fi).sum() * self.alpha
        else:
            aux_loss = 0.0
        return weights.type_as(x), indices, aux_loss

class Expert(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, inter_dim, bias=False)
        self.w2 = nn.Linear(inter_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, inter_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class MoE(nn.Module):
    def __init__(self, dim, moe_inter_dim, n_routed_experts=8, n_activated_experts=2, n_shared_experts=2):
        super().__init__()
        self.dim = dim
        self.n_routed_experts = n_routed_experts
        self.n_activated_experts = n_activated_experts
        self.gate = Gate(dim=dim, topk=n_activated_experts, n_routed_experts=n_routed_experts)
        self.experts = nn.ModuleList([Expert(dim, moe_inter_dim) for i in range(self.n_routed_experts)])
        self.shared_experts =Expert(dim, n_shared_experts * moe_inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shape = x.size()
        x = x.view(-1, self.dim)
        weights, indices, aux_loss = self.gate(x)
        y = torch.zeros_like(x)
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(0, self.n_routed_experts):
            if counts[i] == 0:
                continue
            expert = self.experts[i]
            idx, top = torch.where(indices == i)
            y[idx] += expert(x[idx]) * weights[idx, top, None]
        z = self.shared_experts(x)
        self.aux_loss = aux_loss
        return (y + z).view(shape)

##----------------------------------------------------------------------------------------------------------------------
##----------------------------------------------------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)

class MLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        hidden_dim = int(4 * dim)
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base

    def forward(self, seq_len):
        inv_freq = 1.0 / (self.base ** (
            torch.arange(0, self.dim, 2) / self.dim
        ))
        t = torch.arange(seq_len)
        freqs = torch.einsum("i,j->ij", t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos_emb = emb.cos()[None, None, :, :]
        sin_emb = emb.sin()[None, None, :, :]
        return cos_emb, sin_emb

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.kv_heads = 2
        self.head_dim = dim // n_heads
        self.n_rep = self.n_heads // self.kv_heads
        self.wq = nn.Linear(dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(dim, self.kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, dim, bias=False)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def repeat_kv(self, x):
        batch_size, seq_len, n_kv_heads, head_dim = x.shape
        if self.n_rep == 1:
            return x
        else:
            return (
                x.unsqueeze(3)
                .expand(batch_size, seq_len, n_kv_heads, self.n_rep, head_dim)
                .reshape(batch_size, seq_len, n_kv_heads * self.n_rep, head_dim)
            )

    def apply_rotary(self, x, cos, sin):
        return (x * cos) + (rotate_half(x) * sin)

    def forward(self, x, signal_token_num=0, cos=None, sin=None):
        batch_size, seq_len, _ = x.shape
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device))
        if signal_token_num > 0:
            causal_mask[:signal_token_num, :signal_token_num] = 1
        attn_mask = causal_mask.bool()
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = self.q_norm(xq.view(batch_size, -1, self.n_heads, self.head_dim)).permute(0, 2, 1, 3)
        xk = self.k_norm(xk.view(batch_size, -1, self.kv_heads, self.head_dim))
        xv = xv.view(batch_size, -1, self.kv_heads, self.head_dim)
        xk = self.repeat_kv(xk).permute(0, 2, 1, 3)
        xv = self.repeat_kv(xv).permute(0, 2, 1, 3)
        # Apply RoPE using precomputed cos/sin
        if cos is not None and sin is not None:
            if signal_token_num > 0:
                xq[:, :, :signal_token_num] = self.apply_rotary(xq[:, :, :signal_token_num], cos[:, :, :signal_token_num], sin[:, :, :signal_token_num])
                xk[:, :, :signal_token_num] = self.apply_rotary(xk[:, :, :signal_token_num], cos[:, :, :signal_token_num], sin[:, :, :signal_token_num])
            if signal_token_num < seq_len:
                xq[:, :, signal_token_num:] = self.apply_rotary(xq[:, :, signal_token_num:], cos[:, :, signal_token_num:], sin[:, :, signal_token_num:])
                xk[:, :, signal_token_num:] = self.apply_rotary(xk[:, :, signal_token_num:], cos[:, :, signal_token_num:], sin[:, :, signal_token_num:])

        output = F.scaled_dot_product_attention(
            xq, xk, xv,
            attn_mask=attn_mask,
            is_causal=False
        ).permute(0, 2, 1, 3)
        output = output.flatten(-2)
        return self.wo(output)

class SignalEncoder(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.dim = dim
        self.n_fft = 200
        self.hop_length = self.n_fft // 4
        self.register_buffer("hann_window", torch.hann_window(self.n_fft))
        self.conv_proj = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=dim, kernel_size=(5, 1), stride=(5, 1), bias=False),
            nn.GroupNorm(num_groups=32, num_channels=dim),
            nn.GELU()
        )
        self.channel_norm = RMSNorm(dim)
        self.channel_attn = Attention(dim, n_heads=8)

    def forward(self, x):
        b, c, l = x.shape
        # ----------------------------------------Z-Score---------------------------------------------------------------
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        x_std = torch.std(x, dim=-1, keepdim=True)
        x = (x - x_mean) / (x_std + 1e-6)
        # --------------------------------------------------------------------------------------------------------------
        remainder = 200 * math.ceil(l / 200) - l
        if remainder > 0:
            x = F.pad(x, (0, remainder), mode='constant', value=0)
        # --------------------------------------------------------------------------------------------------------------
        x = x.reshape(b * c, -1)
        x = torch.abs(torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, window=self.hann_window, return_complex=True, normalized=True))
        x = torch.log1p(x)  # 对数压缩
        x = x.unsqueeze(1)
        x = self.conv_proj(x)
        _, d, h, w = x.shape
        # (b c) d h w -> (b h w) c d
        x = rearrange(x, '(b c) d h w -> (b h w) c d', b=b, c=c, d=d, h=h, w=w)
        x = x + self.channel_attn(self.channel_norm(x))
        x = torch.mean(x, dim=1, keepdim=True)
        x = rearrange(x, '(b h w) 1 d -> b (h w) d', b=b, d=d, h=h, w=w)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=8, use_moe=False):
        super(TransformerBlock, self).__init__()
        self.self_attn = Attention(dim=dim, n_heads=num_heads)
        if use_moe:
            self.mlp = MoE(dim=dim, moe_inter_dim=dim)
        else:
            self.mlp = MLP(dim=dim)
        self.input_norm = RMSNorm(dim)
        self.post_attention_layernorm = RMSNorm(dim)

    def forward(self, x, signal_token_num=0, cos=None, sin=None):
        h = x + self.self_attn(self.input_norm(x), signal_token_num, cos, sin)
        out = h + self.mlp(self.post_attention_layernorm(h))
        return out

class Metis(nn.Module):
    def __init__(self, dim=768, vocab_size=151936, num_heads=8, n_layers=12, signal_base=10000, text_base=100000):
        super(Metis, self).__init__()
        self.dim = dim
        self.n_layers = n_layers
        self.Encoder = SignalEncoder(dim=dim)
        self.embedding = nn.Embedding(vocab_size, dim)
        self.block = nn.ModuleList([
            TransformerBlock(dim=dim, num_heads=num_heads, use_moe=False) if i < 2
            else TransformerBlock(dim=dim, num_heads=num_heads, use_moe=True)
            for i in range(self.n_layers)
        ])
        self.signal_rope = RotaryEmbedding(dim // num_heads, base=signal_base)
        self.text_rope = RotaryEmbedding(dim // num_heads, base=text_base)
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)
        self.embedding.weight = self.lm_head.weight

    def forward(self, x, input_ids=None):
        x = self.Encoder(x)
        signal_token_num = x.shape[1]
        device = x.device
        cos, sin = self.signal_rope(signal_token_num)
        cos = cos.to(device)
        sin = sin.to(device)
        if input_ids is not None:
            input_ids = self.embedding(input_ids)
            x = torch.cat([x, input_ids], dim=1)
            cos_text, sin_text = self.text_rope(x.shape[1] - signal_token_num)
            cos_text = cos_text.to(device)
            sin_text = sin_text.to(device)
            cos = torch.cat([cos, cos_text], dim=2)
            sin = torch.cat([sin, sin_text], dim=2)
        for block in self.block:
            x = block(x, signal_token_num=signal_token_num, cos=cos, sin=sin)
        x = self.norm(x)
        if input_ids is not None:
            x = self.lm_head(x[:, signal_token_num:])
        if self.training:
            aux_loss = sum(
                layer.mlp.aux_loss
                for layer in self.block
                if isinstance(layer.mlp, MoE)
            )
            return x, aux_loss
        else:
            return x

class MetisClassifier(nn.Module):
    def __init__(self, dim=768, vocab_size=151936, num_heads=12, n_layers=12, signal_base=10000, text_base=1000000, num_classes=5):
        super(MetisClassifier, self).__init__()
        self.backbone = Metis(dim, vocab_size, num_heads, n_layers, signal_base, text_base)
        self.classification_head = nn.Linear(dim, num_classes)

    def forward(self, x):
        if self.training:
            signal_feature, aux_loss = self.backbone(x)
        else:
            signal_feature = self.backbone(x)
        signal_feature = torch.mean(signal_feature, dim=1, keepdim=False)
        pre = self.classification_head(signal_feature)
        return pre

if __name__ == "__main__":
    model = Metis()
    x = torch.randn(1, 10, 1000)
    z = torch.randint(low=1, high=20000, size=(1, 20))
    print(z.shape)
    y = model(x, z)
    print(y.shape)
