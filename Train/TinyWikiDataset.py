from datasets import load_dataset, concatenate_datasets
from torch.utils.data import Dataset
import torch
from tqdm import tqdm
from langchain.text_splitter import RecursiveCharacterTextSplitter

class TinyWikiDataset(Dataset):
    def __init__(self, tokenizer, block_size=128, max_articles=None):
        self.tokenizer = tokenizer
        self.block_size = block_size
        print("Function called")
        dataset = concatenate_datasets([load_dataset("wikipedia", "20220301.simple", split="train"), load_dataset("wikitext", "wikitext-2-v1", split="train")])
        self.examples = []
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=block_size * 4,  # Larger chunk size to account for tokenization
            chunk_overlap=50,  # Reduced overlap to prevent duplicate content
            length_function=lambda x: len(self.tokenizer.encode(x)),  # Use tokenizer length
        )
        
        for i, article in enumerate(tqdm(dataset, desc="Processing dataset")):
            if max_articles and i >= max_articles:
                break
            
            # Split the text using LangChain's RecursiveCharacterTextSplitter
            chunks = text_splitter.split_text(article['text'])
            
            for chunk in chunks:
                tokenized = self.tokenizer.encode(chunk, truncation=False, add_special_tokens=False)
                
                # Handle chunks that are longer than block_size
                for i in range(0, len(tokenized), block_size):
                    block = tokenized[i:i + block_size]
                    
                    # Pad if necessary
                    if len(block) < block_size:
                        block = block + [self.tokenizer.pad_token_id] * (block_size - len(block))
                    block = [item for item in block if item is not None]
                    self.examples.append(block)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)