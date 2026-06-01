from training import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(123)

model = GPTModel(GPT_CONFIG_124M_TRAINING)
model.to(device)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.0004, weight_decay=0.1
)

num_epochs = 10

train_losses, val_losses, tokens_seen = train_model_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=5, eval_iter=5,
    start_context="Every effort moves you", tokenizer=tokenizer
)



