import torch
import torch.nn as nn

# GPT BLOCK
class GPTBlock(nn.Module):

    def __init__(
        self,
        d_model,
        nhead,
        d_ff,
        dropout=0.1
    ):

        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )

        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        seq_len = x.size(1)

        # Prevent each token from attending
        # to future tokens.
        causal_mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        attn_output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=causal_mask
        )

        x = self.norm1(
            x + self.dropout(attn_output)
        )

        ffn_output = self.ffn(x)

        x = self.norm2(
            x + self.dropout(ffn_output)
        )

        return x

#TINY GPT
class TinyGPT(nn.Module):

    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        nhead,
        n_layers,
        d_ff,
        dropout
    ):

        super().__init__()

        self.token_embedding = nn.Embedding(
            vocab_size,
            d_model
        )

        self.positional_encoding = nn.Embedding(
            max_seq_len,
            d_model
        )

        self.blocks = nn.ModuleList([
            GPTBlock(
                d_model=d_model,
                nhead=nhead,
                d_ff=d_ff,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        self.lm_head = nn.Linear(
            d_model,
            vocab_size
        )

    def forward(self, input_ids):

        batch_size, seq_len = input_ids.shape

        positions = torch.arange(
            seq_len,
            device=input_ids.device
        )

        x = (
            self.token_embedding(input_ids)
            + self.positional_encoding(positions)
        )

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits
