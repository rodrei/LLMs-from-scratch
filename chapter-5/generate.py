import sys
import os
import torch

# Add the 'chapter 4' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../chapter4')))

# Import everything
from main import *
from training import *

torch.manual_seed(123)

token_ids = generate(
    model=model,
    idx=text_to_token_ids("Every effort moves you", tokenizer),
    max_new_tokens=15,
    context_size=GPT_CONFIG_124M_TRAINING["context_length"],
    top_k=25,
    temperature=1.4
)

print("Output text: \n", token_ids_to_text(token_ids, tokenizer))
