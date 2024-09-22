import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from transformers import AutoTokenizer, PreTrainedModel, DataCollatorForLanguageModeling, PretrainedConfig
from datasets import load_dataset
from torch.utils.data import DataLoader
from typing import List
import math
from tqdm import tqdm
import gc

# Configuration for the advanced model
class AdvancedLLMConfig(PretrainedConfig):
    model_type = "advanced_llm"

    def __init__(self, vocab_size=30000, d_model=768, nhead=12, dim_feedforward=3072,
                 max_position_embeddings=2048, memory_size=2000, num_layers=12, eos_token_id=50256, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.max_position_embeddings = max_position_embeddings
        self.memory_size = memory_size
        self.num_layers = num_layers
        self.eos_token_id = eos_token_id

# Positional encoding implementation
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [d_model/2]
        pe[:, 0::2] = torch.sin(position * div_term)  # Apply sinusoidal functions
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :].to(x.dtype)  # Add positional encoding
        return x

# Encoder implementation
class MultiResolutionEncoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, num_layers: int):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure input is 3D: [batch_size, seq_len, d_model]
        assert x.dim() == 3, f"Expected 3D input tensor but got {x.dim()}D tensor"
        return self.encoder(x)

class AdaptiveAttentionDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int):
        super().__init__()
        self.multi_head_attention = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, d_model)
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # Ensure x and memory are correctly shaped
        assert x.dim() == 3 and memory.dim() == 3, f"Expected 3D input tensors but got {x.dim()}D and {memory.dim()}D tensors"
        attended, _ = self.multi_head_attention(x, memory, memory)
        x = self.layer_norm1(x + attended)
        x = self.layer_norm2(x + self.feed_forward(x))
        return x

# Model definition
class AdvancedLLM(PreTrainedModel):
    config_class = AdvancedLLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_position_embeddings)
        self.encoder = MultiResolutionEncoder(config.d_model, config.nhead, config.dim_feedforward, config.num_layers)
        self.decoder = AdaptiveAttentionDecoder(config.d_model, config.nhead, config.dim_feedforward)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids.long())
        x = self.positional_encoding(x)
        encoded = self.encoder(x)
        decoded = self.decoder(x, encoded)  # Use x as query and encoded as memory
        return self.output_layer(decoded)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int, temperature: float = 1.0) -> List[int]:
        self.eval()
        generated = input_ids[0].tolist()  # Assuming batch size is 1
        for _ in range(max_length - len(generated)):
            inputs = torch.tensor([generated]).to(input_ids.device)
            outputs = self(inputs)
            next_token_logits = outputs[0, -1, :] / temperature
            probabilities = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1).item()
            generated.append(next_token)
            if next_token == self.config.eos_token_id:
                break
        return [generated]

def train_model(model, train_dataloader, num_epochs=3, learning_rate=5e-5,
                device='cuda', accumulation_steps=8, max_grad_norm=1.0):
    model.to(device)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()

    for epoch in range(num_epochs):
        total_loss = 0
        optimizer.zero_grad()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for i, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            with autocast('cuda'):
                outputs = model(input_ids)
                loss = criterion(outputs.view(-1, outputs.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        torch.save(model.state_dict(), f'checkpoint_epoch_{epoch + 1}.pt')
        gc.collect()

    return model

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("wikipedia", "20220301.en", split="train[:15000]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']})

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_dataloader = DataLoader(tokenized_dataset, batch_size=5, collate_fn=data_collator, shuffle=True)

    config = AdvancedLLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=384,
        nhead=12,
        dim_feedforward=2304,
        max_position_embeddings=2048,
        memory_size=2000,
        num_layers=12,
        eos_token_id=tokenizer.eos_token_id
    )

    model = AdvancedLLM(config)

    model_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # Updated for float32
    print(f"Model size: {model_size:.2f} MB")

    # After training the model
    trained_model = train_model(model, train_dataloader, num_epochs=5, learning_rate=5e-5, device=device, accumulation_steps=8)

    # Move the model to CPU for saving
    trained_model = trained_model.to('cpu')
    trained_model.save_pretrained("./advanced_llm_model")
    tokenizer.save_pretrained("./advanced_llm_model")
    print("Model training complete and saved in Hugging Face format.")

    # Prepare input for generation
    input_text = "The history of artificial intelligence"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to('cpu')

    # Generate text
    generated_ids = trained_model.generate(input_ids, max_length=50)
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("Generated Text:", generated_text)

if __name__ == "__main__":
    main()