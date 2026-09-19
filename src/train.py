import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .tokenizer import Vocabulary, encode
from .dataset import GPTDataset
from .model import TinyGPT

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

vocab = Vocabulary()

vocab.build(sentences)

token_ids = []

for sentence in sentences:

    ids = encode(sentence,vocab)

    token_ids.extend(ids)   

context_length = 8

dataset = GPTDataset(
    token_ids,
    context_length
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = TinyGPT(
    vocab_size = len(vocab),
    max_seq_len=context_length,
    d_model=128,
    nhead=4,
    n_layers=2,
    d_ff=512,
    dropout=0.1
).to(device)

loss_fn = nn.CrossEntropyLoss()

optimiser = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4
)

epochs = 20

for epoch in range(epochs):

    model.train()

    total_loss = 0.0

    for batch in loader:

        input_ids = batch["input_ids"].to(device)

        labels = batch["labels"].to(device)

        logits = model(input_ids)

        logits = logits.reshape(-1,logits.size(-1))

        labels = labels.reshape(-1)

        loss = loss_fn(logits,labels)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        total_loss += loss.item()

    average_loss = total_loss/len(loader)

    print(
        f"Epoch {epoch + 1}/{epochs}"
        f"Loss: {average_loss:.4f}"
    )   
