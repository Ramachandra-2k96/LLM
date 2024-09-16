from Train.TransformerConfig import TransformerConfig
from transformers import PreTrainedModel, PretrainedConfig, BertTokenizerFast
from Train.TinyWikiDataset import TinyWikiDataset
from Train.TransformerModel import TransformerModel
from Train.Train import train
import torch
import torch.nn as nn 
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

if __name__ == '__main__':
    config = TransformerConfig(
        vocab_size=30522,
        hidden_size=384,            # Model hidden size
        num_attention_heads=12,     # Number of heads
        num_hidden_layers=12,       # Number of layers
        intermediate_size=3072,     # Feed-forward dimension
        dropout=0.05,               # Dropout probability
        max_position_embeddings=1024
    )

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased",clean_up_tokenization_spaces=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    dataset = TinyWikiDataset(tokenizer, block_size=config.max_position_embeddings)
    model = TransformerModel(config)
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    model.apply(init_weights)
    model = torch.compile(model)
    trained_model = train(model, tokenizer, dataset, batch_size=3, learning_rate=2e-4, num_epochs=3, config=config)
    
    path = "tinywiki_model_Think"
    trained_model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"Model and tokenizer saved at {path}")