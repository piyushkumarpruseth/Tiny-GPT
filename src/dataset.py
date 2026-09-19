import torch
from torch.utils.data import Dataset

#GPT DATASET
class GPTDataset(Dataset):

    def __init__(self, token_ids, context_length):

        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self):

        return len(self.token_ids) - self.context_length

    def __getitem__(self, idx):

        input_ids = self.token_ids[
            idx:idx + self.context_length
        ]

        labels = self.token_ids[
            idx + 1:idx + self.context_length + 1
        ]

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long
            )
        }
