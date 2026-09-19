# TinyGPT

A small GPT-style autoregressive language model built from scratch with **PyTorch**.

This project is designed as a learning project to understand the main components behind decoder-only Transformer language models rather than relying entirely on high-level libraries.

## Architecture

```text
Text
  ↓
Whitespace Tokenisation
  ↓
Vocabulary
  ↓
Token IDs
  ↓
Token Embedding + Positional Embedding
  ↓
Causal Self-Attention
  ↓
Feed-Forward Network (GELU)
  ↓
Layer Normalisation
  ↓
Linear Language Model Head
  ↓
Next-token prediction
```

## Project structure

```text
tiny-gpt/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── model.py
│   ├── tokenizer.py
│   ├── dataset.py
│   ├── train.py
│   └── generate.py
├── examples/
│   └── output.txt
└── checkpoints/
    └── .gitkeep
```

## Requirements

- Python 3.10+
- PyTorch

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run training

From the project root:

```bash
python -m src.train
```

The script trains TinyGPT on a deliberately tiny toy dataset and prints the training loss.

## Generate text

After training:

```bash
python -m src.generate
```

The generation script loads the trained checkpoint and generates text from a starting prompt.

## Model configuration

The default model uses:

| Parameter | Value |
|---|---:|
| Embedding dimension | 128 |
| Attention heads | 4 |
| Transformer blocks | 2 |
| Feed-forward dimension | 512 |
| Context length | 8 |
| Dropout | 0.1 |
| Learning rate | 3e-4 |
| Epochs | 20 |

## Example dataset

The initial dataset contains six simple sentences:

```text
the cat sat on the mat
the cat is happy
the dog sat on the mat
the dog is happy
the cat likes food
the dog likes food
```

This dataset is intentionally tiny. The goal is to understand the implementation, not to train a useful language model.

## Key concepts demonstrated

- Vocabulary construction
- Special tokens: `<PAD>`, `<UNK>`, `<BOS>`, `<EOS>`
- Token-to-ID and ID-to-token conversion
- Next-token prediction
- Causal self-attention
- Transformer residual connections
- Layer normalization
- GELU activation
- Cross-entropy language-model loss
- AdamW optimization
- Autoregressive text generation

## Important implementation detail

The attention layer uses a **causal mask**, so a token cannot attend to future tokens during training.

For example:

```text
Token 1 → sees 1
Token 2 → sees 1, 2
Token 3 → sees 1, 2, 3
Token 4 → sees 1, 2, 3, 4
```

This prevents the model from seeing the answer while learning to predict the next token.

## Limitations

This is an educational TinyGPT, not a production language model.

Limitations include:

- Extremely small training dataset
- Very small vocabulary
- Simple whitespace tokenisation
- No validation/test split
- No batching across independently separated documents
- Greedy decoding using `argmax`
- No checkpoint management beyond the basic saved model
- No distributed training
- No large-scale dataset or pretraining

## Future improvements

Possible next steps:

1. Add a larger text dataset.
2. Implement a better tokenizer.
3. Add train/validation loss tracking.
4. Add temperature and top-k sampling.
5. Add model checkpointing.
6. Add configurable hyperparameters.
7. Add tests for tokenisation, dataset creation, and generation.
8. Compare the implementation with a Hugging Face Transformer model.
9. Add experiment tracking.
10. Build a small web interface for generation.

## Learning goal

The main purpose of this repository is to understand how a GPT-style model works internally:

> tokens → embeddings → causal attention → feed-forward layers → logits → next token

It is intentionally small enough to read and understand the complete implementation.

## License

This project is provided for educational purposes. Add a license appropriate to your intended use before publishing if needed.
# Tiny-GPT
