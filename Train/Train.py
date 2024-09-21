from torch.nn.utils.rnn import pad_sequence
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader

def collate_fn(batch, pad_token_id):
    # Sort the batch in descending order of length
    batch.sort(key=lambda x: len(x), reverse=True)
    sequences = batch  # No need for list comprehension as items are already tensors
    
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
    
    return padded_sequences


def train(model, tokenizer, dataset, batch_size, learning_rate, num_epochs, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        pad_token_id = tokenizer.pad_token_id

    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=lambda batch: collate_fn(batch, pad_token_id),
        num_workers=4,  # Increased for faster data loading
        pin_memory=True
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    
    # Scheduler with warm-up and adaptive learning rate reduction
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=learning_rate, 
                                                    steps_per_epoch=len(dataloader),
                                                    epochs=num_epochs, pct_start=0.1)
    
    scaler = torch.amp.GradScaler('cuda')

    # Adaptive accumulation steps - decrease over time for larger effective batch sizes
    init_accumulation_steps = 32
    final_accumulation_steps = 8
    accumulation_steps_decay = (init_accumulation_steps - final_accumulation_steps) // num_epochs

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        accumulation_steps = init_accumulation_steps - (accumulation_steps_decay * epoch)

        for i, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            batch = batch.to(device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(batch)['last_hidden_state']
                logits = model.lm_head(outputs)  # Add this to get the logits for language modeling
                logits = logits[:, :-1, :].contiguous()
                targets = batch[:, 1:].contiguous()

                attention_mask = (batch != pad_token_id).float()
                attention_mask = attention_mask[:, 1:].contiguous()

                loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1),
                                    ignore_index=pad_token_id, reduction='none')
                loss = (loss * attention_mask.view(-1)).sum() / attention_mask.sum()

                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    continue

            loss = loss / accumulation_steps
            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()
                
                torch.cuda.empty_cache()

            total_loss += loss.item() * accumulation_steps

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        #if (epoch + 1) % 3 == 0:
        checkpoint_path = f"checkpoint_epoch_{epoch + 1}"
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    return model