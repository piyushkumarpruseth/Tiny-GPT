import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .tokenizer import Vocabulary, encode
from .dataset import GPTDataset
from .model import TinyGPT


# =========================
# DATASET
# =========================

text = """
cat is an animal
dog is an animal
cat likes milk
dog likes bones
cat likes to play
dog likes to play
the cat is small
the dog is big
cat and dog are friends
"""

sentences = text.strip().split("\n")


# =========================
# VOCABULARY
# =========================

vocab = Vocabulary()
vocab.build(sentences)


# =========================
# TOKENISE DATASET
# =========================

token_ids = []

for sentence in sentences:

    ids = encode(sentence, vocab)

    token_ids.extend(ids)


# =========================
# CONFIGURATION
# =========================

context_length = 8

d_model = 128
nhead = 4
n_layers = 2
d_ff = 512
dropout = 0.1

learning_rate = 3e-4
epochs = 20
batch_size = 4


# =========================
# DATASET + DATALOADER
# =========================

dataset = GPTDataset(
    token_ids,
    context_length
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)


# =========================
# DEVICE
# =========================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# =========================
# MODEL
# =========================

model = TinyGPT(
    vocab_size=len(vocab),
    max_seq_len=context_length,
    d_model=d_model,
    nhead=nhead,
    n_layers=n_layers,
    d_ff=d_ff,
    dropout=dropout
).to(device)


# =========================
# LOSS + OPTIMIZER
# =========================

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# =========================
# TRAINING
# =========================

print("\nStarting training...\n")

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for batch in loader:

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        logits = model(input_ids)

        logits = logits.reshape(
            -1,
            logits.size(-1)
        )

        labels = labels.reshape(-1)

        loss = loss_fn(
            logits,
            labels
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {average_loss:.4f}"
    )


# =========================
# SAVE CHECKPOINT
# =========================

checkpoint = {
    "model_state_dict": model.state_dict(),
    "vocab_token_to_id": vocab.token_to_id,
    "vocab_id_to_token": vocab.id_to_token,
    "context_length": context_length,
    "d_model": d_model,
    "nhead": nhead,
    "n_layers": n_layers,
    "d_ff": d_ff,
    "dropout": dropout
}

torch.save(
    checkpoint,
    "tinygpt_checkpoint.pt"
)

print("\nModel saved to tinygpt_checkpoint.pt")
