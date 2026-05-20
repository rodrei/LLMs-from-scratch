import torch
import torch.nn as nn
import tiktoken
from main import GPTModel, GPT_CONFIG_124M

tokenizer = tiktoken.get_encoding("gpt2")

batch = []

txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))

batch = torch.stack(batch, dim=0)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)


out = model(batch)


total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")







