import torch
import torch.nn as nn
import tiktoken
from main import LayerNorm

torch.manual_seed(123)

batch_example = torch.randn(2,5)

ln = LayerNorm(emb_dim=5)
out_ln = ln(batch_example)

mean = out_ln.mean(dim=-1, keepdim=True)
var = out_ln.var(dim=-1, unbiased=False, keepdim=True)

torch.set_printoptions(sci_mode=False)

print("Mean:\n", mean)
print("Variance:\n", var)
