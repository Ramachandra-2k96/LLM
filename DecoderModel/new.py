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
import os

class AdvancedLLMConfig(PretrainedConfig):
    model_type = "advanced_llm"

    def __init__(
        self,
        vocab_size=30000,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        max_position_embeddings=512,
        memory_size=1000,
        num_layers=6,
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
        self.short_range = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mid_range = nn.TransformerEncoder(encoder_layer, num_layers)
        self.long_range = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fusion = nn.Linear(d_model * 3, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        short = self.short_range(x)
        mid = self.mid_range(x)
        long = self.long_range(x)
        return self.fusion(torch.cat([short, mid, long], dim=-1))

class AdaptiveAttentionDecoder(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, max_seq_len: int = 512):
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
        self.memory_bank = nn.Parameter(torch.randn(memory_size, d_model))
        self.context_selector = nn.MultiheadAttention(d_model, 8, batch_first=True)
        self.memory_updater = nn.GRUCell(d_model, d_model)
        self.d_model = d_model
        self.memory_size = memory_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        batch_size = x.size(0)
        expanded_memory = self.memory_bank.unsqueeze(0).expand(batch_size, -1, -1)
        
        context, _ = self.context_selector(x, expanded_memory, expanded_memory)
        context_mean = context.mean(dim=1)
        
        updated_memories = []
        for i in range(batch_size):
            updated_memory = self.memory_updater(context_mean[i], self.memory_bank[0].clone())
            updated_memories.append(updated_memory)
        
        updated_memory = torch.stack(updated_memories)
        self.memory_bank.data = updated_memory.mean(dim=0).unsqueeze(0)
        
        return context

class AdvancedLLM(PreTrainedModel):
    config_class = AdvancedLLMConfig

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.max_position_embeddings)
        self.encoder = MultiResolutionEncoder(config.d_model, config.nhead, config.dim_feedforward, config.num_layers)
        self.memory = ContextAwareMemory(1, config.d_model) 
        self.decoder = AdaptiveAttentionDecoder(config.d_model, config.nhead, config.dim_feedforward)
        self.output_layer = nn.Linear(config.d_model, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        encoded = self.encoder(x)
        memory = self.memory(encoded)
        decoded = self.decoder(encoded, memory)
        return self.output_layer(decoded)

    def generate(self, input_ids: torch.Tensor, max_length: int, temperature: float = 1.0) -> List[int]:
        self.eval()
        generated = input_ids.tolist()
        
        with torch.no_grad():
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
                num_epochs: int, learning_rate: float, device: torch.device):
    model.to(device)
    model.train()

    # Ensure all model parameters are float32
    for param in model.parameters():
        param.data = param.data.to(torch.float32)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            with autocast(device_type='cuda', dtype=torch.float32):
                output = model(input_ids)
                loss = criterion(output.view(-1, output.size(-1)), labels.view(-1))

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint
        model.save_pretrained(f'checkpoint_epoch_{epoch + 1}')

    return model

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Load and prepare the Wikipedia dataset
    dataset = load_dataset("wikipedia", "20220301.en", split="train[:100]")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, max_length=512, padding="max_length")

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=dataset.column_names)
    tokenized_dataset = tokenized_dataset.map(lambda examples: {'labels': examples['input_ids']})

    # Prepare the data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Create the dataloader
    train_dataloader = DataLoader(tokenized_dataset, batch_size=2, collate_fn=data_collator)

    # Initialize the configuration and model
    config = AdvancedLLMConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        nhead=8,
        dim_feedforward=2048,
        max_position_embeddings=512,
        memory_size=1000,
        num_layers=6,
        eos_token_id=tokenizer.eos_token_id
    )

    model = AdvancedLLM(config)
    for param in model.parameters():
        param.data = param.data.to(torch.float32)
    # Train the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trained_model = train_model(model, train_dataloader, num_epochs=3, learning_rate=1e-4, device=device)

    # Save the model in Hugging Face format
    trained_model.save_pretrained("./advanced_llm_model")
    tokenizer.save_pretrained("./advanced_llm_model")

    print("Model training complete and saved in Hugging Face format.")

    # Example of text generation
    input_text = "The history of artificial intelligence"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    generated_ids = trained_model.generate(input_ids[0], max_length=100, temperature=0.7)
    generated_text = tokenizer.decode(generated_ids)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()