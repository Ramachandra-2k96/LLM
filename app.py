from Train.TransformerModel import TransformerModel
from transformers import BertTokenizerFast
import torch
model_path = "fine_tuned_custom_model"
model = TransformerModel.from_pretrained(model_path)
tokenizer = BertTokenizerFast.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def generate_prompt(model, tokenizer, prompt, max_length=50, context_window_size=512, temperature=1.0, top_k=50, top_p=1.0):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    input_ids = input_ids[:, -context_window_size:]
    generated_ids = model.generate(
        input_ids, 
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        repetition_penalty=2.0,
        do_sample=True
    )
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

while(True):
    test_prompt = input("User => ")
    if test_prompt.lower() == 'exit':
        break
    print("AI => "+generate_prompt(model, tokenizer, "Question: "+test_prompt+" Answer: "))