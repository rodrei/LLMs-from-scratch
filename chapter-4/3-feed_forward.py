import torch
import torch.nn as nn
import tiktoken
from main import FeedForward, GPT_CONFIG_124M

ffn = FeedForward(GPT_CONFIG_124M)

# [batch_size, num_token, emb_size]
x = torch.rand(2, 3, 768)
out = ffn(x)
print(out.shape)
