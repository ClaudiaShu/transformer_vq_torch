import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, LayerNorm, Linear
from torch.nn.init import xavier_uniform_
import dataclasses
from dataclasses import dataclass

import torch
import torch.nn as nn
import dataclasses
from typing import Any, Callable, List

# Type aliases for PyTorch
PRNGKey = Any
Shape = List[int]
Dtype = torch.dtype
Initializer = Callable[[Shape, Dtype], torch.nn.Module]

def sg(x):
    return x.detach()

def st(x):
    return x - sg(x)

def get_sinusoid_embs(length, width, lam, flip, start=0):
    pos_seq = torch.arange(start, start + length)
    inv_lams = 1 / (lam ** (torch.arange(0, width, 2) / width))
    pre = pos_seq.unsqueeze(-1) * inv_lams
    sin = torch.sin(pre)
    cos = torch.cos(pre)
    cat = torch.cat([sin, cos], dim=-1)
    
    if flip:
        cat = torch.flip(cat, dims=[0])
    
    return cat

def get_shortcodes(vecs, codebook):
    B, H, L, K = vecs.shape
    S = codebook.shape[1]
    
    assert vecs.shape == (B, H, L, K)
    assert codebook.shape == (H, S, K)

    vecs_sq = torch.sum(vecs ** 2, dim=-1, keepdim=True)  # B, H, L, 1
    codebook_sq = torch.sum(codebook ** 2, dim=-1, keepdim=False)  # H, S

    # Compute distance matrix
    diffs2 = (
        vecs_sq
        - 2.0 * torch.einsum('bhlk,hsk->bhls', vecs, codebook)
        + codebook_sq.unsqueeze(0).unsqueeze(2)
    )  # B, H, L, S
    
    z = torch.argmin(diffs2, dim=-1)  # B, H, L
    
    errs2 = torch.min(diffs2, dim=-1).values
    errs2 = torch.relu(errs2)  # Ensure non-negative values
    
    return z.long(), errs2

def get_codewords(shortcodes, codebook):
    B, H, L = shortcodes.shape
    S, d = codebook.shape[1:]
    i = 1

    shortcodes = shortcodes[..., None]
    codebook = codebook[None, ...]

    assert shortcodes.shape == (B, H, L, i)
    assert codebook.shape == (i, H, S, d)

    code_indices = shortcodes.expand(-1, -1, -1, d)
    cz = torch.gather(codebook, dim=2, index=code_indices)
    return cz

@dataclasses.dataclass
class TransformerConfig:
    param_dtype: Dtype
    dtype: Dtype
    global_batch_size: int
    sequence_len: int
    update_len: int
    block_len: int
    mem_len: int
    grad_thru_cache: bool
    agg_cache: bool
    d_model: int
    d_k: int
    d_v: int
    d_ff: int
    n_head: int
    n_code: int
    n_layer: int
    n_vocab: int
    pe_abs: bool
    pe_lam: float
    p_dropemb: float
    p_dropsin: float
    p_dropres: float
    p_droplyr: float
    p_nucleus: float
    c_beta: float
    c_gamma: float
    e_tie: bool
    e_preln: bool
    e_scale: str
    is_train: bool
    e_init: Initializer
    w_init: Initializer
    r_init: Initializer
    b_init: Initializer
    no_emb: bool = False

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in dataclasses.fields(TransformerConfig)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}

        if isinstance(filtered.get("param_dtype"), str):
            filtered["param_dtype"] = torch.dtype(filtered["param_dtype"])
        if isinstance(filtered.get("dtype"), str):
            filtered["dtype"] = torch.dtype(filtered["dtype"])

        for k, v in filtered.items():
            if signature[k] is bool and v in {0, 1}:
                filtered[k] = bool(v)

        # Set default initializers
        filtered["e_init"] = nn.init.normal_(torch.empty(1), mean=0.0, std=1.0)
        filtered["w_init"] = lambda shape, dtype: nn.init.kaiming_normal_(torch.empty(shape, dtype=dtype), a=0.0, mode='fan_in', nonlinearity='linear')
        filtered["r_init"] = lambda shape, dtype: nn.init.kaiming_normal_(torch.empty(shape, dtype=dtype), a=0.0, mode='fan_in', nonlinearity='linear')
        filtered["b_init"] = nn.init.zeros_(torch.empty(1))

        return cls(**filtered)
    
@dataclass
class VQSpec:
    n_device: torch.Tensor
    n_block_per_update: torch.Tensor
    loss_mask: torch.Tensor

    @classmethod
    def create(cls, **kwargs):
        signature = {field.name: field.type for field in dataclasses.fields(cls)}
        filtered = {k: v for k, v in kwargs.items() if k in signature}
        return cls(**filtered)

class ScaledSin(nn.Module):
    def __init__(self, config):
        super(ScaledSin, self).__init__()
        self.config = config
        self.apply_config()
        self.scale = nn.Parameter(torch.tensor(self.b_init, dtype=torch.float32))

    def apply_config(self):
        for k, v in vars(self.config).items():
            setattr(self, k, v)

    def get_sinusoid_embeddings(self, length, offset):
        # Generates sinusoidal embeddings
        position = torch.arange(offset, offset + length, dtype=torch.float32)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * -(np.log(10000.0) / self.d_model))
        embeddings = torch.zeros(length, self.d_model)
        embeddings[:, 0::2] = torch.sin(position[:, None] * div_term[None, :])
        embeddings[:, 1::2] = torch.cos(position[:, None] * div_term[None, :])
        return embeddings

    def forward(self, length, offset):
        embs = self.get_sinusoid_embeddings(length=length, offset=offset)
        return self.scale * embs
    
class Embeddings(nn.Module):
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.config = config
        self.apply_config()
        
        self.embs = nn.Parameter(torch.randn(self.n_vocab, self.d_model) * self.e_init)
        
        self.bias_out = nn.Parameter(torch.randn(self.n_vocab) * self.b_init)

    def apply_config(self):
        for k, v in vars(self.config).items():
            setattr(self, k, v)

    def forward(self, x):
        # Embedding lookup
        x = self.embs[x]  # Assumes x contains indices
        return x

    def logits(self, x):
        x = x.float()  # Ensure x is in float type for matrix multiplication
        x = torch.matmul(x, self.embs.t())  # Compute logits
        x += self.bias_out  # Add bias term
        return x

class LearnableVQ(nn.Module):
    def __init__(self, config):
        super(LearnableVQ, self).__init__()
        self.config = config
        self.apply_config()

        # Initialize parameters
        cs_args = (self.w_init, (self.n_head, self.n_code, self.d_k), self.param_dtype)
        cc_args = (self.ones_init, (self.n_head, self.n_code), self.param_dtype)
        self.c_sum = nn.Parameter(self.init_tensor(*cs_args))  # shape HSd
        self.c_count = nn.Parameter(self.init_tensor(*cc_args))  # shape HS

    def apply_config(self):
        for k, v in vars(self.config).items():
            setattr(self, k, v)

    def ones_init(self, shape, dtype):
        return torch.ones(shape, dtype=dtype)

    def init_tensor(self, init_fn, shape, dtype):
        return init_fn(shape, dtype)

    @staticmethod
    def _get_codebook(c_sum, c_count):
        # Avoid division by zero by using a small epsilon value
        epsilon = 1e-2
        c_count = torch.clamp(c_count, min=epsilon)
        c = c_sum / c_count.unsqueeze(-1)
        return c.detach()

    def get_codebook(self):
        return LearnableVQ._get_codebook(self.c_sum, self.c_count)

    @staticmethod
    def get_codebook_ema_targets(vecs, shortcodes, c_sum, c_count, c_gamma, vq_spec):
        n_code = c_sum.shape[1]
        B, H, L, d = vecs.shape
        S = c_sum.shape[1]
        
        r = F.one_hot(shortcodes, num_classes=n_code).to(vecs.dtype)
        loss_mask = vq_spec.loss_mask
        if loss_mask.dim() == 3:  
            loss_mask = loss_mask.squeeze(1)
        else:
            loss_mask = loss_mask
        r *= loss_mask.unsqueeze(1).unsqueeze(-1)
        
        c_sum_hat = vq_spec.n_device * vq_spec.n_block_per_update * torch.einsum("bhts,bhtd->hsd", r, vecs)
        c_count_hat = vq_spec.n_device * vq_spec.n_block_per_update * torch.sum(r, dim=(0, 2))
        
        c_sum_tgt = (1 - c_gamma) * c_sum_hat + c_gamma * c_sum
        c_count_tgt = (1 - c_gamma) * c_count_hat + c_gamma * c_count
        
        return c_sum_tgt, c_count_tgt

    @staticmethod
    def get_codebook_loss(vecs, shortcodes, c_sum, c_count, c_gamma, vq_spec):

        c_sum_tgt, c_count_tgt = LearnableVQ.get_codebook_ema_targets(
            vecs=vecs,
            shortcodes=shortcodes,
            c_sum=c_sum,
            c_count=c_count,
            c_gamma=c_gamma,
            vq_spec=vq_spec,
        )
        
        l_codebook_sum = torch.sum((c_sum - c_sum_tgt).detach() * c_sum)
        l_codebook_count = torch.sum((c_count - c_count_tgt).detach() * c_count)
        l_codebook = l_codebook_count + l_codebook_sum
        
        return l_codebook

    @staticmethod
    def get_quantization_metrics(vecs, vecs_hat, errs2, c_sum, c_count, dtype):
        n_head, n_code = c_count.shape[0], c_count.shape[1]
        eps, errmin, errmax, maskval = 1e-2, 0e1, 1e1, 1e30
        
        c_count = torch.clamp(c_count, min=eps)
        c = c_sum / c_count[..., None]
        
        c_norms = torch.clamp(torch.norm(c, dim=-1), min=eps)
        c_normed = c / c_norms[..., None]
        
        # Cosine similarity and pairwise distances
        c_sims = torch.einsum("hsd,hzd->hsz", c_normed, c_normed)
        c_dists = torch.norm(c.unsqueeze(2) - c.unsqueeze(1), dim=-1)
        
        vec_norms = torch.clamp(torch.norm(vecs, dim=-1), min=eps)
        vec_hat_norms = torch.clamp(torch.norm(vecs_hat, dim=-1), min=eps)
    
        errs = torch.sqrt(errs2)
        relative_errs = torch.clamp(errs / vec_norms, min=errmin, max=errmax)
        
        probs = c_count / torch.sum(c_count, dim=-1, keepdim=True)
        c_thresh_oob = (c_count < 1.0) | (c_count > 1_000_000)
        c_thresh_oob = c_thresh_oob.float()
        
        # Upper and lower triangular masks
        ones = torch.ones([1, n_code, n_code], dtype=torch.float32, device=c_count.device)
        up = torch.triu(ones)
        low = torch.tril(ones, diagonal=-1)
        
        metrics = dict(
            c_sim_min=torch.min(low * c_sims + maskval * up),
            c_sim_mean=torch.sum(low * c_sims, dim=(1, 2)) / torch.sum(low, dim=(1, 2)),
            c_sim_max=torch.max(low * c_sims - maskval * up),
            c_dist_min=torch.min(low * c_dists + maskval * up),
            c_dist_mean=torch.sum(low * c_dists, dim=(1, 2)) / torch.sum(low, dim=(1, 2)),
            c_dist_max=torch.max(low * c_dists - maskval * up),
            c_norm_min=torch.min(c_norms, dim=1).values,
            c_norm_mean=torch.mean(c_norms, dim=1),
            c_norm_max=torch.max(c_norms, dim=1).values,
            c_usage_min=torch.min(c_count, dim=1).values,
            c_usage_mean=torch.mean(c_count, dim=1),
            c_usage_max=torch.max(c_count, dim=1).values,
            c_thresh_oob=torch.sum(c_thresh_oob, dim=1),
            c_entropy=torch.sum(-probs * torch.log(probs + 1e-10), dim=-1),
            vec_norm_mean=torch.mean(vec_norms, dim=2),
            vec_hat_norm_mean=torch.mean(vec_hat_norms, dim=2),
            relative_err_min=torch.min(relative_errs, dim=2).values,
            relative_err_mean=torch.mean(relative_errs, dim=2),
            relative_err_max=torch.max(relative_errs, dim=2).values,
        )
        
        # Convert metrics to specified dtype
        metrics = {k: v.to(dtype) for k, v in metrics.items()}
        
        return metrics

    def forward(self, vecs, vq_spec):
        orig_dtype = vecs.dtype
        vecs_hp = vecs.to(self.param_dtype)
        c = LearnableVQ._get_codebook(self.c_sum, self.c_count)  # build codebook
        z, errs2 = get_shortcodes(vecs=vecs_hp, codebook=c)
        errs2 = errs2.to(self.dtype)
        cz = get_codewords(shortcodes=z, codebook=c)
        cz = cz.to(orig_dtype)
        vecs_hat = cz.detach() + vecs
        
        if self.is_train:
            loss_mask = vq_spec.loss_mask
            l_commit = torch.mean(torch.sum(loss_mask.unsqueeze(1) * errs2, dim=1))
            l_codebook = LearnableVQ.get_codebook_loss(
                vecs=vecs_hp,
                shortcodes=z,
                c_sum=self.c_sum,
                c_count=self.c_count,
                c_gamma=self.c_gamma,
                vq_spec=vq_spec,
            ).to(self.dtype)
        else:
            l_commit = torch.tensor(0.0, dtype=self.dtype)
            l_codebook = torch.tensor(0.0, dtype=self.dtype)

        if self.is_train:
            metrics = LearnableVQ.get_quantization_metrics(
                vecs=vecs.detach(),
                vecs_hat=vecs_hat.detach(),
                errs2=errs2.detach(),
                c_sum=self.c_sum.detach(),
                c_count=self.c_count.detach(),
                dtype=self.dtype,
            )
        else:
            metrics = dict()

        return dict(
            quantized=vecs_hat,
            shortcodes=z,
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metrics,
            errs2=errs2,
        )

class VQAttention(nn.Module):
    def __init__(self, config):
        super(VQAttention, self).__init__()
        self.config = config
        self.apply_config()
        
        self.tau = self.config.d_k**0.5
        
        self.input_ln = nn.LayerNorm(self.config.d_model)
        self.q_ln = nn.LayerNorm(self.config.d_k, elementwise_affine=False)
        self.k_ln = nn.LayerNorm(self.config.d_k, elementwise_affine=False)
        
        q_ch = self.config.n_head * self.config.d_k
        k_ch = self.config.n_head * self.config.d_k
        v_ch = self.config.n_head * self.config.d_v
        
        self.q_proj = nn.Linear(self.config.d_model, q_ch, bias=False)
        self.kvg_proj = nn.Linear(self.config.d_model, k_ch + v_ch + v_ch, bias=False)
        self.r_proj = nn.Linear(self.config.d_model, k_ch, bias=False)
        self.res_proj = nn.Linear(self.config.d_v, self.config.d_model, bias=False)
        
        self.xl_u = nn.Parameter(torch.randn(q_ch))
        self.xl_v = nn.Parameter(torch.randn(q_ch))
        
        self.quantizer = LearnableVQ(self.config)
        
        self.dropsin = nn.Dropout(self.config.p_dropsin)
        self.dropres = nn.Dropout(self.config.p_dropres)

    def apply_config(self):
        # Iterate over the fields of the dataclass and set them as attributes
        for k, v in dataclasses.asdict(self.config).items():
            setattr(self, k, v)
    
    @staticmethod
    def initial_state(config, batch_size):
        prefix = [batch_size, config.n_head]
        s = config.n_code
        m = config.mem_len
        d_k = config.d_k
        d_v = config.d_v
        
        return {
            "pos_offset": torch.tensor(0, dtype=torch.int32),
            "xlcache": {
                "z": torch.full([*prefix, m], fill_value=s, dtype=torch.int32),
                "k_hat": torch.zeros([*prefix, m, d_k], dtype=torch.float32),
                "v": torch.zeros([*prefix, m, d_v], dtype=torch.float32),
                "doc_ids": torch.zeros([batch_size, m], dtype=torch.int32)
            },
            "aggcache": {
                "upper_div_lower": torch.zeros([*prefix, s, d_v], dtype=torch.float32),
                "lower": torch.zeros([*prefix, s], dtype=torch.float32),
                "latest_doc_id": torch.zeros([batch_size], dtype=torch.int32)
            }
        }

    def rel_shift(x):
        *leading_shape, present_len, past_len = x.shape
        x = torch.nn.functional.pad(x, (0, 1, 0, 0))
        x = x.view(*leading_shape, past_len + 1, present_len)
        x = x[..., 1:, :]
        x = x.view(*leading_shape, present_len, past_len)
        return x

    @staticmethod
    def get_causal_mask(block_len, mem_len, invalid_len, with_locality):
        i = torch.arange(block_len).unsqueeze(-1)
        j = torch.arange(mem_len + block_len).unsqueeze(0)
        alloc_mask = j >= invalid_len
        causal_mask = j <= i
        window_mask = j >= i
        keep_mask = alloc_mask & causal_mask
        if with_locality:
            keep_mask = keep_mask & window_mask
        return keep_mask

    @staticmethod
    def get_agg_biases(lower):
        result = torch.where(
            lower == 0,
            -float('inf'),
            torch.log(torch.clamp(lower, min=1.0))
        )
        return result

    def get_q(self, x_tilde):
        if x_tilde.dim() == 4:  
            x_tilde = x_tilde.squeeze(1)
        else:
            x_tilde = x_tilde
        bsz, present_len, _ = x_tilde.shape
        q = self.q_proj(x_tilde)
        q = q.view(bsz, present_len, self.config.n_head, self.config.d_k)
        q = self.q_ln(q) * (self.tau**-0.5)
        q = q.transpose(1, 2)
        return q

    def get_kvg(self, x_tilde):
        if x_tilde.dim() == 4:  
            x_tilde = x_tilde.squeeze(1)
        else:
            x_tilde = x_tilde
        bsz, present_len, *_ = x_tilde.shape
        kvg = self.kvg_proj(x_tilde)
        splits = [self.config.d_k, self.config.d_v, kvg.shape[-1]-self.config.d_k-self.config.d_v]
        kvg_splits = torch.split(kvg, splits, dim=-1)
        
        k, v, g = kvg_splits
        assert k.shape == (bsz, present_len, self.config.n_head * self.config.d_k)
        assert v.shape == (bsz, present_len, self.config.n_head * self.config.d_v)
        assert g.shape == (bsz, present_len, self.config.n_head * self.config.d_v)
        
        k = k.view(bsz, present_len, self.config.n_head, self.config.d_k)
        v = v.view(bsz, present_len, self.config.n_head, self.config.d_v)
        
        k = self.k_ln(k) * (self.tau**-0.5)
        
        v = F.silu(v)
        g = F.silu(g)
        
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)
        return k.to(dtype=self.config.param_dtype), v, g

    def get_xl_helpers(self):
        xl_r = get_sinusoid_embs(
            length=self.config.mem_len + self.config.block_len,
            width=self.config.d_model,
            lam=self.config.pe_lam,
            flip=True
        )
        xl_r = self.dropsin(xl_r)
        xl_r = self.r_proj(xl_r)
        xl_r = xl_r.view(self.config.mem_len + self.config.block_len, self.config.n_head, self.config.d_k)
        xl_r = xl_r.transpose(0, 1)
        xl_r = xl_r * (self.tau**-0.5)
        xl_u = self.xl_u.view(1, self.config.n_head, 1, self.config.d_k) * (self.tau**-0.5)
        xl_v = self.xl_v.view(1, self.config.n_head, 1, self.config.d_k) * (self.tau**-0.5)
        return xl_r, xl_u, xl_v

    def attn(self, present_q, present_k, present_v, present_doc_ids, state, vq_spec):
        bsz = present_q.shape[0]
        dims = {
            'B': bsz,
            'L': self.config.block_len,
            'M': self.config.mem_len,
            'W': self.config.mem_len + self.config.block_len,
            'H': self.config.n_head,
            'S': self.config.n_code,
            'K': self.config.d_k,
            'V': self.config.d_v
        }

        if present_doc_ids.dim() == 3:  
            present_doc_ids = present_doc_ids.squeeze(1)
        else:
            present_doc_ids = present_doc_ids
        
        vq_output_dict = self.quantizer(present_k, vq_spec=vq_spec)
        present_z = vq_output_dict["shortcodes"]
        present_k_hat = vq_output_dict["quantized"]
        l_commit = vq_output_dict["l_commit"]
        l_codebook = vq_output_dict["l_codebook"]
        metrics = vq_output_dict["metrics"]

        xlcache = state["xlcache"]
        aggcache = state["aggcache"]
        recent_z = torch.cat([xlcache["z"], present_z], dim=-1)
        recent_k_hat = torch.cat([xlcache["k_hat"], present_k_hat], dim=-2)
        recent_v = torch.cat([xlcache["v"], present_v], dim=-2)
        recent_doc_ids = torch.cat([xlcache["doc_ids"], present_doc_ids], dim=-1)

        xl_r, xl_u, xl_v = self.get_xl_helpers()

        c = self.quantizer.get_codebook()
        cache_scores = torch.einsum("bhlk,hsk->bhls", present_q + xl_u, c)
        cache_biases = VQAttention.get_agg_biases(aggcache["lower"])
        cache_biases = cache_biases.unsqueeze(-2)
        cache_scores += cache_biases

        recent_scores_ac = torch.einsum("bhlk,bhwk->bhlw", present_q + xl_u, recent_k_hat)
        recent_scores_bd = torch.einsum("bhlk,hwk->bhlw", present_q + xl_v, xl_r)
        recent_scores_bd = VQAttention.rel_shift(recent_scores_bd)
        causal_mask = VQAttention.get_causal_mask(
            block_len=self.config.block_len,
            mem_len=self.config.mem_len,
            invalid_len=max(0, self.config.mem_len - state["pos_offset"].item()),
            with_locality=True
        ).to(torch.int32)
        recent_scores_bd *= causal_mask[None, None, ...]
        recent_scores = recent_scores_ac + recent_scores_bd
        keep_mask = VQAttention.get_causal_mask(
            block_len=self.config.block_len,
            mem_len=self.config.mem_len,
            invalid_len=max(0, self.config.mem_len - state["pos_offset"].item()),
            with_locality=not self.config.agg_cache
        ).to(torch.int32)
        recent_scores = recent_scores * keep_mask
        
        # Subtract max score for stability
        cache_max_scores = torch.max(cache_scores, dim=-1)[0]
        recent_max_scores = torch.max(recent_scores, dim=-1)[0]
        max_scores = torch.maximum(cache_max_scores, recent_max_scores)
        cache_scores -= max_scores.unsqueeze(-1)
        recent_scores -= max_scores.unsqueeze(-1)
        cache_a = torch.exp(cache_scores)
        recent_a = torch.exp(recent_scores)

        d = torch.sum(recent_a, dim=-1)
        if self.config.agg_cache:
            d += torch.sum(cache_a, dim=-1)
        wv = torch.einsum("bhlw,bhwv->bhlv", recent_a / d.unsqueeze(-1), recent_v)
        if self.config.agg_cache:
            wv += torch.einsum("bhls,bhsv->bhlv", cache_a / d.unsqueeze(-1), aggcache["upper_div_lower"])

        wv = wv.transpose(1, 2)
        wv = wv.view(bsz, self.config.block_len, self.config.n_head * self.config.d_v)
        
        return {
            "attn_out": wv,
            "recent_z": recent_z,
            "recent_k_hat": recent_k_hat,
            "recent_v": recent_v,
            "recent_doc_ids": recent_doc_ids,
            "l_commit": l_commit,
            "l_codebook": l_codebook,
            "metrics": metrics
        }

    def update_state(self, recent_z, recent_k_hat, recent_v, recent_doc_ids, state):
        bsz = recent_z.shape[0]
        aggcache = state["aggcache"]
        new_pos_offset = state["pos_offset"] + self.config.block_len

        clamped_recent_z = torch.clamp(recent_z[..., :-self.config.mem_len], min=0, max=self.config.n_code - 1)
        one_hot_z = F.one_hot(clamped_recent_z, num_classes=self.config.n_code).float()

        new_lower = aggcache["lower"] + torch.sum(one_hot_z, dim=-2)
        f1 = aggcache["lower"] / torch.clamp(new_lower, min=1.0)
        f2 = one_hot_z / torch.clamp(new_lower.unsqueeze(-2), min=1.0)
        
        new_upper_div_lower = f1[..., None] * aggcache["upper_div_lower"] + torch.einsum(
            "bhls,bhlv->bhsv", f2, recent_v[..., :-self.config.mem_len, :]
        )

        new_state = {
            "pos_offset": new_pos_offset,
            "xlcache": {
                "z": recent_z[..., -self.config.mem_len :],
                "k_hat": recent_k_hat[..., -self.config.mem_len :, :],
                "v": recent_v[..., -self.config.mem_len :, :],
                "doc_ids": recent_doc_ids[..., -self.config.mem_len :]
            },
            "aggcache": {
                "lower": new_lower,
                "upper_div_lower": new_upper_div_lower,
                "latest_doc_id": recent_doc_ids[..., -self.config.mem_len - 1]
            }
        }
        
        return new_state

    def forward(self, state, input_dict):
        doc_ids = input_dict.pop("doc_ids")
        vq_spec = input_dict.pop("vq_spec")
        x = input_dict.pop("input_features")
        x_tilde = self.input_ln(x)
        q = self.get_q(x_tilde=x_tilde)
        k, v, g = self.get_kvg(x_tilde=x_tilde)
        attn_output_dict = self.attn(q, k, v, doc_ids, state, vq_spec)
        wv = attn_output_dict["attn_out"]
        o = wv * g
        res = self.res_proj(o)
        res = self.dropres(res)

        new_state = self.update_state(
            recent_z=attn_output_dict["recent_z"],
            recent_k_hat=attn_output_dict["recent_k_hat"],
            recent_v=attn_output_dict["recent_v"],
            recent_doc_ids=attn_output_dict["recent_doc_ids"],
            state=state
        )
        
        output_dict = {
            "res": res,
            "metrics": attn_output_dict["metrics"],
            "l_commit": attn_output_dict["l_commit"],
            "l_codebook": attn_output_dict["l_codebook"]
        }
        
        return new_state, output_dict

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super(TransformerLayer, self).__init__()
        self.config = config
        self.apply_config()

        self.scanned_attn1 = VQAttention(config)
        self.scanned_attn2 = VQAttention(config)
        
        self.droplyr1 = Dropout(self.p_droplyr)
        self.droplyr2 = Dropout(self.p_droplyr)

    def apply_config(self):
        for k, v in vars(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def initial_state(config, batch_size):
        return [
            VQAttention.initial_state(config=config, batch_size=batch_size),
            VQAttention.initial_state(config=config, batch_size=batch_size),
        ]

    def forward(self, x, doc_ids, state, vq_spec):
        batch_size, *_ = x.shape
        dims = {
            'B': batch_size,
            'L': self.block_len,
            'D': self.d_model,
        }
        
        state1, state2 = state

        assert x.shape == (dims['B'], dims['L'], dims['D'])
        
        attn1_input_dict = {'input_features': x, 'doc_ids': doc_ids, 'vq_spec': vq_spec}
        attn1_state, attn1_output_dict = self.scanned_attn1(state1, attn1_input_dict)
        r1 = attn1_output_dict.pop("res")
        assert r1.shape == (dims['B'], dims['L'], dims['D'])
        x += self.droplyr1(r1)

        attn2_input_dict = {'input_features': x, 'doc_ids': doc_ids, 'vq_spec': vq_spec}
        attn2_state, attn2_output_dict = self.scanned_attn2(state2, attn2_input_dict)
        r2 = attn2_output_dict.pop("res")
        assert r2.shape == (dims['B'], dims['L'], dims['D'])
        x += self.droplyr2(r2)

        l_commit = attn1_output_dict.pop("l_commit") + attn2_output_dict.pop("l_commit")
        l_codebook = attn1_output_dict.pop("l_codebook") + attn2_output_dict.pop("l_codebook")
        metric_dict = {k: (v + attn2_output_dict["metrics"][k]) / 2 for k, v in attn1_output_dict["metrics"].items()}

        return dict(
            output_features=x,
            attn_state=[attn1_state, attn2_state],
            l_commit=l_commit,
            l_codebook=l_codebook,
            metrics=metric_dict,
        )
    
class Transformer(nn.Module):
    def __init__(self, config):
        super(Transformer, self).__init__()
        self.config = config
        self.apply_config()

        if not self.no_emb or self.e_tie:
            self.token_embedder = Embeddings(self.config)
        if self.pe_abs:
            self.position_embedder = ScaledSin(self.config)
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(self.config) for _ in range(self.n_layer)
        ])
        if self.e_preln:
            self.out_ln = LayerNorm(self.d_model)
        if not self.e_tie:
            self.out_proj = Linear(
                self.d_model, 
                self.n_vocab, 
                bias=True
            )
        self.dropemb = Dropout(self.p_dropemb)

    def apply_config(self):
        for k, v in vars(self.config).items():
            setattr(self, k, v)

    @staticmethod
    def initial_state(config, batch_size):
        return [
            TransformerLayer.initial_state(
                config=config,
                batch_size=batch_size,
            )
            for _ in range(config.n_layer)
        ]

    def get_chex_dims(self, batch_size, present_len):
        return {
            'B': batch_size,
            'P': present_len,
            'K': present_len // self.block_len,
            'L': self.block_len,
            'D': self.d_model,
            'V': self.n_vocab,
        }

    @staticmethod
    def maybe_aggregate(accumulator_dict, new_dict):
        if not accumulator_dict:
            return new_dict
        if not new_dict:
            return accumulator_dict
        # return {k: a + b for k, (a, b) in zip(accumulator_dict.keys(), zip(accumulator_dict.values(), new_dict.values()))}
        # Recursive function to handle nested dictionaries and apply element-wise addition
        def dict_tree_map(func, dict1, dict2):
            if isinstance(dict1, dict) and isinstance(dict2, dict):
                # Ensure that both dictionaries have the same keys
                if set(dict1.keys()) != set(dict2.keys()):
                    raise ValueError("Dictionaries must have the same keys")
                
                return {key: dict_tree_map(func, dict1[key], dict2[key]) for key in dict1}
            elif isinstance(dict1, torch.Tensor) and isinstance(dict2, torch.Tensor):
                return func(dict1, dict2)
            else:
                raise TypeError("Unsupported types for dict_tree_map")

        # Define the addition function for tensors
        def add_tensors(tensor1, tensor2):
            return tensor1 + tensor2

        # Apply the function using the dict_tree_map
        return dict_tree_map(add_tensors, accumulator_dict, new_dict)

    @staticmethod
    def average_layer_metrics(aux, n_layer):
        if "metrics" not in aux:
            return aux
        metrics = aux.pop("metrics")
        metrics = {k: v / n_layer for k, v in metrics.items()}
        new_aux = {'metrics': metrics, **aux}
        return new_aux

    def forward(self, inputs, doc_ids, state, vq_spec):
        batch_size, present_len, *_ = inputs.shape
        dims = self.get_chex_dims(batch_size, present_len)
        assert doc_ids.shape == (dims['B'], dims['P'])
        new_state = []
        aux = {}
        x = inputs
        if not self.no_emb:
            x = self.token_embedder(x)
        if self.pe_abs:
            offset = state[0][0]["pos_offset"]
            x += self.position_embedder(length=present_len, offset=offset)
        x = self.dropemb(x)
        assert x.shape == (dims['B'], dims['L'], dims['D'])
        for i in range(self.n_layer):
            layer_output_dict = self.transformer_layers[i](
                x=x, doc_ids=doc_ids, state=state[i], vq_spec=vq_spec
            )
            new_state.append(layer_output_dict.pop("attn_state"))
            x = layer_output_dict.pop("output_features")
            assert x.shape == (dims['B'], dims['L'], dims['D'])
            aux = Transformer.maybe_aggregate(aux, layer_output_dict)

        aux = Transformer.average_layer_metrics(aux, self.n_layer)
        if self.e_preln:
            x = self.out_ln(x)
        x = self.token_embedder.logits(x) if self.e_tie else self.out_proj(x)
        x *= self.e_scale
        x = F.log_softmax(x, dim=-1)
        assert x.shape == (dims['B'], dims['P'], dims['V'])
        return dict(logprobs=x, attn_state=new_state, **aux)


# Define the configuration for the Transformer model
config = {
    'param_dtype': torch.float32,
    'no_emb': False,
    'pe_abs': False,
    'n_vocab': 2,
    'd_model': 768,
    'n_head': 1,
    'd_k': 128,
    'd_v': 1536,
    'n_code': 512,
    'block_len': 16,
    'mem_len': 16,
    'n_layer': 24,
    'p_droplyr': 0.1,
    'p_dropemb': 0.1,
    'p_dropsin': 0.1,
    'p_dropres': 0.1,
    'p_dropscale': 0.1,
    'p_dropattn': 0.1,
    'p_dropff': 0.1,
    'p_dropact': 0.1,
    'p_droplyr': 0.1,
    'p_drophead': 0.1,
    'p_dropblock': 0.1,
    'p_dropatt': 0.1,
    'p_dropmem': 0.1,
    'p_dropvq': 0.1,
    'p_dropq': 0.1,
    'p_dropk': 0.1,
    'p_dropv': 0.1,
    'p_dropg': 0.1,
    'p_dropxl': 0.1,
    'p_dropcache': 0.1,
    'p_dropupper': 0.1,
    'p_droplower': 0.1,
    'p_dropdoc': 0.1,
    'dtype': torch.float32,
    'is_train': True,
    'global_batch_size': 1,
    'sequence_len': 4096,
    'block_len': 16,
    'update_len': 2048,
    'grad_thru_cache': False,
    'agg_cache': False,
    'd_ff': 0,
    'pe_lam': 1.0,
    'p_nucleus': 0.9,
    'c_beta': 0.25,
    'c_gamma': 0.99,
    'e_tie': True,
    'e_preln': True,
    'e_scale': 1.0
    }

configs = TransformerConfig.create(**config)

# Create the model
model = Transformer(configs)

# Define the input tensors
inputs = torch.zeros([1, configs.block_len], dtype=torch.int32)
doc_ids = torch.zeros([1, configs.block_len], dtype=torch.int32)
state = Transformer.initial_state(config=configs, batch_size=1)
vq_spec = VQSpec.create(
    n_device=torch.tensor([1]),
    n_block_per_update=torch.tensor([1]),
    loss_mask=torch.ones([1, configs.block_len], dtype=torch.int32)
)

# Perform a forward pass of the model
output = model(inputs, doc_ids, state, vq_spec)
print(output.keys())
# dict_keys(['logprobs', 'attn_state', 'metrics', 'l_commit', 'l_codebook'])
print(output['logprobs'].shape)
print(output['attn_state'][0][0]['xlcache']['z'].shape)


