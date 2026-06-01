import sys
import os
import torch

# Add the 'chapter 4' directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../chapter4')))

# Import everything
from main import *
from dataloader import *


GPT_CONFIG_124M_TRAINING = {
    "vocab_size": 50257,
    "context_length": 256,    #1
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "drop_rate": 0.1,       #2
    "qkv_bias": False
}

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M_TRAINING)
model.eval() # Disable random components like dropout


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

# start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")


# SAMPLE 1
#
#token_ids = generate_text_simple(
#    model=model,
#    idx=text_to_token_ids(start_context, tokenizer),
#    max_new_tokens=10,
#    context_size=GPT_CONFIG_124M_TRAINING["context_length"]
#)
#
# print("Output text:\n", token_ids_to_text(token_ids, tokenizer))



# SAMPLE 2

#inputs = torch.tensor([[16833, 3626, 6100],   # ["every effort moves",
#                       [40,    1107, 588]])   #  "I really like"]
#
#targets = torch.tensor([[3626, 6100, 345  ],  # [" effort moves you",
#                        [1107, 588, 11311]])  #  " really like chocolate"]”
#
#
#with torch.no_grad():     #1
#    logits = model(inputs)
#probas = torch.softmax(logits, dim=-1)     #2
#
#print("probas shape: ", probas.shape)
#
#token_ids = torch.argmax(probas, dim=-1, keepdim=True)
#print("Token IDs:\n", token_ids)
#
#
#print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
#print(f"Outputs batch 1:"
#      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
#
#text_idx = 0
#target_probas_1 = probas[text_idx, [0, 1, 2], targets[text_idx]]
#print("Text 1:", target_probas_1)
#
#text_idx = 1
#target_probas_2 = probas[text_idx, [0, 1, 2], targets[text_idx]]
#print("Text 2:", target_probas_2)


file_path = "the-verdict.txt"
with open(file_path, "r", encoding="utf-8") as file:
    text_data = file.read()


total_characters = len(text_data)
total_tokens = len(tokenizer.encode(text_data))
print("Characters:", total_characters)
print("Tokens:", total_tokens)

train_ratio = 0.90
split_idx = int(train_ratio * len(text_data))
train_data = text_data[:split_idx]
val_data = text_data[split_idx:]


train_loader = create_dataloader_v1(
    train_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M_TRAINING["context_length"],
    stride=GPT_CONFIG_124M_TRAINING["context_length"],
    drop_last=True,
    shuffle=True,
    num_workers=0
)
val_loader = create_dataloader_v1(
    val_data,
    batch_size=2,
    max_length=GPT_CONFIG_124M_TRAINING["context_length"],
    stride=GPT_CONFIG_124M_TRAINING["context_length"],
    drop_last=False,
    shuffle=False,
    num_workers=0
)

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)         #1
    target_batch = target_batch.to(device)      
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):

    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, "
                    f"Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen

def evaluate_model(model, train_loader, val_loder, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )

    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_emb.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer)

    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )

    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()



