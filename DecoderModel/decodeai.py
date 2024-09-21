import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from transformers import AutoTokenizer, PreTrainedModel, DataCollatorForLanguageModeling, PretrainedConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import List
import math
from tqdm import tqdm
import gc

class AdvancedLLMConfig(PretrainedConfig):
    model_type = "advanced_llm"

    def __init__(
        self,
        vocab_size=30000,
        d_model=768,
        nhead=12,
        dim_feedforward=3072,
        max_position_embeddings=2048,
        memory_size=2000,
        num_layers=12,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.max_position_embeddings = max_position_embeddings
        self.memory_size = memory_size
        self.num_layers = num_layers

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class MultiResolutionEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

class AdaptiveAttentionDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, max_seq_len: int = 2048):
        super().__init__()
        self.nhead = nhead
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.attention_modulator = nn.Linear(d_model, nhead * max_seq_len)
        self.multi_head_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        nhead = self.nhead

        attention_weights = self.attention_modulator(x)
        attention_weights = attention_weights[:, :, :nhead * seq_len]
        attention_weights = attention_weights.view(batch_size, seq_len, nhead, seq_len)
        attention_weights = attention_weights.permute(0, 2, 1, 3)
        attention_weights = attention_weights.reshape(batch_size * nhead, seq_len, seq_len)

        attended, _ = self.multi_head_attention(x, memory, memory, attn_mask=attention_weights)
        x = self.layer_norm1(x + attended)
        x = self.layer_norm2(x + self.feed_forward(x))
        return x

class ContextAwareMemory(nn.Module):
    def __init__(self, memory_size: int, d_model: int):
        super().__init__()
        self.memory_bank = nn.Parameter(torch.randn(1, memory_size, d_model))
        self.context_selector = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.memory_updater = nn.GRUCell(d_model, d_model)
        self.d_model = d_model
        self.memory_size = memory_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.size()
        
        # Expand memory_bank to match batch size
        expanded_memory = self.memory_bank.expand(batch_size, -1, -1)
        
        context, _ = self.context_selector(x, expanded_memory, expanded_memory)
        context_mean = context.mean(dim=1)
        
        updated_memories = []
        for i in range(batch_size):
            updated_memory = self.memory_updater(context_mean[i], self.memory_bank[0, 0].clone())
            updated_memories.append(updated_memory)
        
        updated_memory = torch.stack(updated_memories)
        self.memory_bank = nn.Parameter(updated_memory.unsqueeze(1))
        
        return context

class AdvancedLLM(PreTrainedModel):
    config_class = AdvancedLLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_position_embeddings)
        self.encoder = MultiResolutionEncoder(config.d_model, config.nhead, config.dim_feedforward, config.num_layers)
        self.memory = ContextAwareMemory(config.memory_size, config.d_model)
        self.decoder = AdaptiveAttentionDecoder(config.d_model, config.nhead, config.dim_feedforward)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)
        memory = self.memory(encoded)
        decoded = self.decoder(encoded, memory)
        return self.output_layer(decoded)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int, temperature: float = 1.0) -> List[int]:
        self.eval()
        generated = input_ids.tolist()
        
        for _ in range(max_length - len(generated)):
            inputs = torch.tensor(generated).unsqueeze(0).to(input_ids.device)
            outputs = self(inputs)
            next_token_logits = outputs[0, -1, :] / temperature
            next_token = torch.multinomial(torch.softmax(next_token_logits, dim=-1), num_samples=1).item()
            generated.append(next_token)
            if next_token == self.config.eos_token_id:
                break
        
        return generated

def train_model(model: AdvancedLLM, train_dataloader: DataLoader, 
                num_epochs: int, learning_rate: float, device: torch.device,
                accumulation_steps: int = 32, max_grad_norm: float = 1.0):
    model.to(device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        
        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with autocast('cuda'):
                output = model(input_ids)
                loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * accumulation_steps
            progress_bar.set_postfix({"Loss": loss.item() * accumulation_steps})

            del input_ids, labels, output, loss
            torch.cuda.empty_cache()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        model.cpu()
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch + 1}.pt')
        model.to(device)
        gc.collect()
        torch.cuda.empty_cache()

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:50000]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']})

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(tokenized_dataset, batch_size=6, collate_fn=data_collator, shuffle=True)

    config = AdvancedLLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        nhead=8,
        dim_feedforward=2304,
        max_position_embeddings=2048,
        memory_size=2000,
        num_layers=8,
        eos_token_id=tokenizer.eos_token_id
    )

    model = AdvancedLLM(config)
    
    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # Size in MB
    print(f"Model size: {model_size:.2f} MB")

    trained_model = train_model(model, train_dataloader, num_epochs=3, learning_rate=5e-5, device=device, accumulation_steps=8)

    trained_model = trained_model.to('cpu')
    trained_model.save_pretrained("./advanced_llm_model")
    tokenizer.save_pretrained("./advanced_llm_model")
    print("Model training complete and saved in Hugging Face format.")

    input_text = "The history of artificial intelligence"
    input_ids = tokenizer.encode(input_text, return_tensors="pt")
    generated_ids = trained_model.generate(input_ids[0], max_length=100, temperature=0.7)
    generated_text = tokenizer.decode(generated_ids)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()