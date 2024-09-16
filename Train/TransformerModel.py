import torch.nn as nn
import torch
import torch.nn.functional as F
from transformers import PreTrainedModel
from Train.TransformerConfig import TransformerConfig
from Train.TransformerEmbeddings import TransformerEmbeddings
from Train.TransformerLayer import TransformerLayer
from Train.top_k_top_p_filtering import top_k_top_p_filtering
from Train.ReflectionBlock import ReflectionBlock

class TransformerModel(PreTrainedModel):
    config_class = TransformerConfig
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.embeddings = TransformerEmbeddings(config)
        self.encoder = nn.ModuleList([TransformerLayer(config) for _ in range(config.num_hidden_layers)])
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()
        self.reflection = ReflectionBlock(config)


        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model."""
        for layer, heads in heads_to_prune.items():
            self.encoder[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
        )

        encoder_outputs = embedding_output
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        for i, layer_module in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (encoder_outputs,)

            layer_outputs = layer_module(
                encoder_outputs,
                attention_mask=extended_attention_mask,
                output_attentions=output_attentions,
            )
            encoder_outputs = layer_outputs[0]
            encoder_outputs = self.reflection(encoder_outputs)
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (encoder_outputs,)

        pooled_output = self.pooler_activation(self.pooler(encoder_outputs[:, 0]))

        if not return_dict:
            return tuple(
                v
                for v in [
                    encoder_outputs,
                    pooled_output,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )

        return {
            "last_hidden_state": encoder_outputs,
            "pooler_output": pooled_output,
            "hidden_states": all_hidden_states,
            "attentions": all_self_attentions,
        }
        
    def generate(self, input_ids, max_length=50, temperature=0.8, top_k=50, top_p=0.95, repetition_penalty=1.2, do_sample=True):
        self.eval()
        device = input_ids.device
        input_ids = input_ids.to(device)
        
        generated = input_ids
        past_key_values = None
        
        for _ in range(max_length):
            outputs = self(input_ids=generated)
            logits = self.lm_head(outputs['last_hidden_state'])
            next_token_logits = logits[:, -1, :]

            next_token_logits = next_token_logits / temperature
            
            if repetition_penalty != 1.0:
                for batch_idx in range(next_token_logits.size(0)):
                    for prev_token in set(generated[batch_idx].tolist()):
                        next_token_logits[batch_idx, prev_token] /= repetition_penalty
            
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            
            if do_sample:
                probs = F.softmax(filtered_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(filtered_logits, dim=-1, keepdim=True)
            
            generated = torch.cat((generated, next_token), dim=1)
            
            if next_token.item() == self.config.eos_token_id:
                break
        
        return generated
    
