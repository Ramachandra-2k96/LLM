from transformers import PretrainedConfig

class TransformerConfig(PretrainedConfig):
    def __init__(self,
                 num_reflection_steps=3, 
                 vocab_size=30522,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 num_key_value_heads=4,  # New parameter for GQA
                 intermediate_size=3072,
                 hidden_act="swiglu",  # Changed to SwiGLU
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02,
                 layer_norm_eps=1e-12,
                 pad_token_id=0,
                 bos_token_id=1,
                 eos_token_id=2,
                 max_grad_norm=1.0,
                 position_embedding_type="rotary",  # Changed to rotary
                 use_cache=True,
                 classifier_dropout=None,
                 **kwargs):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.max_grad_norm =max_grad_norm
        self.classifier_dropout = classifier_dropout
        self.num_reflection_steps = num_reflection_steps