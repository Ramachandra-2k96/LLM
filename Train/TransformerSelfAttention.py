import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from typing import Optional
<<<<<<< HEAD
from torch.amp import autocast
from Train.RotaryPositionalEmbedding import RotaryPositionalEmbedding, apply_rotary_pos_emb

=======
from torch.cuda.amp import autocast
from Train.RotaryPositionalEmbedding import RotaryPositionalEmbedding, apply_rotary_pos_emb
>>>>>>> d9dfcb65baab1ed650b437da008dac69dac8056c
try:
    from flash_attn import flash_attn_func
    FLASH_ATTENTION_AVAILABLE = True
except ImportError:
    FLASH_ATTENTION_AVAILABLE = False
    print("Flash Attention not available. Falling back to standard attention.")

class TransformerSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = getattr(config, 'num_key_value_heads', config.num_attention_heads)
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.num_key_value_heads * self.attention_head_size, bias=False)

        self.dropout = config.attention_probs_dropout_prob
        self.rotary_emb = RotaryPositionalEmbedding(self.attention_head_size, config.max_position_embeddings)

    def transpose_for_scores(self, x: torch.Tensor, num_heads: int) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (num_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    @torch.compiler.allow_in_graph
    def flash_attn_wrapper(self, q, k, v):
        return flash_attn_func(
            q, k, v,
            dropout_p=self.dropout,
            softmax_scale=1.0 / math.sqrt(self.attention_head_size),
            causal=True
        )

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
<<<<<<< HEAD
        with autocast('cuda'):
=======
        with autocast(enabled=True):
>>>>>>> d9dfcb65baab1ed650b437da008dac69dac8056c
            query_layer = self.transpose_for_scores(self.query(hidden_states), self.num_attention_heads)
            key_layer = self.transpose_for_scores(self.key(hidden_states), self.num_key_value_heads)
            value_layer = self.transpose_for_scores(self.value(hidden_states), self.num_key_value_heads)

            seq_len = query_layer.shape[2]
            cos, sin = self.rotary_emb(value_layer, seq_len)
            query_layer, key_layer = apply_rotary_pos_emb(query_layer, key_layer, cos, sin)

            # Grouped Query Attention
            key_layer = torch.repeat_interleave(key_layer, self.num_attention_heads // self.num_key_value_heads, dim=1)
            value_layer = torch.repeat_interleave(value_layer, self.num_attention_heads // self.num_key_value_heads, dim=1)

            if FLASH_ATTENTION_AVAILABLE:
                # Flash Attention expects (batch_size, seq_len, num_heads, head_dim)
                query_layer = query_layer.transpose(1, 2)
                key_layer = key_layer.transpose(1, 2)
                value_layer = value_layer.transpose(1, 2)

<<<<<<< HEAD
                attn_output = self.flash_attn_wrapper(query_layer, key_layer, value_layer)
=======
                attn_output = flash_attn_func(
                    query_layer, key_layer, value_layer,
                    dropout_p=self.dropout,
                    softmax_scale=1.0 / math.sqrt(self.attention_head_size),
                    causal=True
                )
>>>>>>> d9dfcb65baab1ed650b437da008dac69dac8056c

                # Transpose back to (batch_size, num_heads, seq_len, head_dim)
                attn_output = attn_output.transpose(1, 2)
            else:
                # Standard attention as fallback
                attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
                attention_scores = attention_scores / math.sqrt(self.attention_head_size)
                
                if attention_mask is not None:
                    attention_scores = attention_scores + attention_mask

                attention_probs = F.softmax(attention_scores, dim=-1)
                attention_probs = F.dropout(attention_probs, p=self.dropout, training=self.training)
                attn_output = torch.matmul(attention_probs, value_layer)

            attn_output = attn_output.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = attn_output.size()[:-2] + (self.all_head_size,)
            attn_output = attn_output.view(*new_context_layer_shape)

        outputs = (attn_output,)
        if output_attentions:
            outputs += (attention_probs,) if not FLASH_ATTENTION_AVAILABLE else (None,)
        return outputs