import torch
import torch.nn as nn
import tiktoken
from main import *

tokenizer = tiktoken.get_encoding("gpt2")

start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)

print("encoded_tensor.shape:", encoded_tensor.shape)

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

model.eval()

out = generate_text_simple(
    model=model,
    idx=encoded_tensor,
    max_new_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"]
)

print("Output:", out)
print("Output length:", len(out[0]))


