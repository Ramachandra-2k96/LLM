import os
import torch
import json
from torch.utils.data import Subset
from datetime import datetime
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import autocast, GradScaler  # Import for mixed precision

def collate_fn(batch, pad_token_id):
    # Sort the batch in descending order of length
    batch.sort(key=lambda x: len(x), reverse=True)
    sequences = batch  # No need for list comprehension as items are already tensors
    
    # Pad sequences
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=pad_token_id)
    
    return padded_sequences

def incremental_train(model, tokenizer, dataset, batch_size, learning_rate, epochs_per_increment, config, checkpoint_dir):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Load training state if it exists
    training_state_path = os.path.join(checkpoint_dir, "training_state.json")
    if os.path.exists(training_state_path):
        with open(training_state_path, "r") as f:
            training_state = json.load(f)
        start_index = training_state["current_index"]
        total_epochs = training_state["total_epochs"]
    else:
        start_index = 0
        total_epochs = 0

    # Define the size of each increment (e.g., 10% of the dataset)
    increment_size = len(dataset) // 10

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_per_increment)
    
    # Set up gradient scaler for mixed precision
    scaler = GradScaler()

    for increment in range(start_index, len(dataset), increment_size):
        end_index = min(increment + increment_size, len(dataset))
        increment_dataset = Subset(dataset, range(increment, end_index))
        
        dataloader = torch.utils.data.DataLoader(
            increment_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            collate_fn=lambda batch: collate_fn(batch, tokenizer.pad_token_id),
            num_workers=4,
            pin_memory=True
        )

        for epoch in range(epochs_per_increment):
            model.train()
            total_loss = 0

            for batch in tqdm(dataloader, desc=f"Epoch {total_epochs + epoch + 1}"):
                batch = batch.to(device)
                
                optimizer.zero_grad()
                
                # Mixed precision training using autocast
                with autocast('cuda'):
                    outputs = model(batch)
                    logits = model.lm_head(outputs['last_hidden_state'])
                    logits = logits[:, :-1, :].contiguous()
                    targets = batch[:, 1:].contiguous()
                
                    loss = F.cross_entropy(logits.view(-1, config.vocab_size), targets.view(-1),
                                           ignore_index=tokenizer.pad_token_id)
                
                # Backward pass with scaled gradients
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)

                # Optimizer step with gradient scaler
                scaler.step(optimizer)
                scaler.update()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Increment {increment//increment_size + 1}, Epoch {total_epochs + epoch + 1}, Average Loss: {avg_loss:.4f}")
            
            scheduler.step()

        total_epochs += epochs_per_increment

        # Save checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_increment_{increment//increment_size + 1}")
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)

        # Update and save training state
        training_state = {
            "current_index": end_index,
            "total_epochs": total_epochs,
            "last_update": datetime.now().isoformat()
        }
        with open(training_state_path, "w") as f:
            json.dump(training_state, f)

        print(f"Checkpoint saved at {checkpoint_path}")

    return model
