import torch.nn as nn
from Train.TransformerSelfAttention import TransformerSelfAttention

class TransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = TransformerSelfAttention(config)
        self.intermediate = nn.Linear(config.hidden_size, config.intermediate_size)
        self.output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.activation = nn.GELU()

    def forward(self, hidden_states, attention_mask=None, output_attentions=False, past_key_value=None):
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions)
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:] 

        attention_output = self.layernorm1(hidden_states + attention_output)
        outputs = self_attention_outputs[1:] 
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.activation(intermediate_output)
        layer_output = self.output(intermediate_output)
        layer_output = self.dropout(layer_output)
        layer_output = self.layernorm2(attention_output + layer_output)

        outputs = (layer_output,) + outputs
        return outputs