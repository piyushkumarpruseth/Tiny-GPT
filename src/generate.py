import torch

from .model import TinyGPT
from .tokenizer import Vocabulary, encode, decode

def generate(
    model,
    input_ids,
    max_new_tokens,
    context_length):

    model.eval()

    for _ in range(max_new_tokens):

        input_ids_cond = input_ids[:,-context_length:]

        with torch.no_grad():

            logits = model(input_ids_cond)

        next_token_logits = logits[:,-1,:]

        next_token = torch.argmax(
            next_token_logits,
            dim=-1
        )

        next_token = next_token.unsqueeze(1)

        input_ids = torch.cat([input_ids,next_token],dim=1)

    return input_ids    

start_text = "The Cat"

start_ids = encode(start_text,vocab)

start_ids = start_ids[:-1]

input_tensor = torch.tensor(
    [start_ids],
    dtype = torch.long,
    device = device
)

generated = generate(
    model,
    input_tensor,
    max_new_tokens = 5,
    context_length = context_length
)

print("Generated IDs:" , generated)

print("Generated text:" , decode(
    generated[0].tolist(),
    vocab
))
