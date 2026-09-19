#TOKENIZER
def tokenise(text):
    return text.lower().split()

#VOCABULARY
class Vocabulary:

    def __init__(self):

        self.token_to_id = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        self.id_to_token = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>"
        }

    def build(self, sentences):

        for sentence in sentences:

            tokens = tokenise(sentence)

            for token in tokens:

                if token not in self.token_to_id:

                    idx = len(self.token_to_id)

                    self.token_to_id[token] = idx
                    self.id_to_token[idx] = token

    def __len__(self):
        return len(self.token_to_id)


def encode(text, vocab):

    tokens = tokenise(text)

    ids = [
        vocab.token_to_id.get(
            token,
            vocab.token_to_id["<UNK>"]
        )
        for token in tokens
    ]

    return (
        [vocab.token_to_id["<BOS>"]]
        + ids
        + [vocab.token_to_id["<EOS>"]]
    )


def decode(ids, vocab):

    tokens = []

    for idx in ids:

        token = vocab.id_to_token[int(idx)]

        if token not in ["<PAD>", "<BOS>", "<EOS>"]:
            tokens.append(token)

    return " ".join(tokens)
