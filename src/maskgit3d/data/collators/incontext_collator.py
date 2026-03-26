"""Variable-length sequence collator with in-context masking."""

import random
from dataclasses import dataclass

import torch


@dataclass
class VariableLengthInContextCollator:
    """Collator for batching variable-length sequences with per-sample masking.

    Handles padding, masking, and creates attention masks and normalized mask weights.
    """

    pad_token_id: int
    mask_token_id: int
    ignore_index: int = -100
    min_mask_ratio: float = 0.1
    max_mask_ratio: float = 0.5

    def __call__(self, batch: list[dict]) -> dict[str, torch.Tensor]:
        """Collate a batch of variable-length sequences.

        Args:
            batch: List of dicts with 'sequence', 'target_mask', and optional 'mask_ratio'.

        Returns:
            Dict with 'input_ids', 'labels', 'attention_mask', 'mask_weights'.

        Raises:
            ValueError: If batch is empty.
        """
        if not batch:
            raise ValueError("Cannot collate empty batch")

        max_seq_len = max(sample["sequence"].shape[0] for sample in batch)
        batch_size = len(batch)

        input_ids = torch.full((batch_size, max_seq_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((batch_size, max_seq_len), self.ignore_index, dtype=torch.long)
        attention_mask = torch.zeros((batch_size, max_seq_len), dtype=torch.long)
        mask_weights = torch.zeros((batch_size, max_seq_len), dtype=torch.float)

        for i, sample in enumerate(batch):
            seq = sample["sequence"]
            target_mask = sample["target_mask"]
            seq_len = seq.shape[0]

            mask_ratio = sample.get("mask_ratio")
            if mask_ratio is None:
                mask_ratio = random.uniform(self.min_mask_ratio, self.max_mask_ratio)

            input_ids[i, :seq_len] = seq.clone()
            attention_mask[i, :seq_len] = 1

            valid_positions = target_mask.nonzero(as_tuple=True)[0]
            num_valid = valid_positions.shape[0]

            if num_valid > 0:
                num_to_mask = max(1, int(num_valid * mask_ratio))
                num_to_mask = min(num_to_mask, num_valid)

                perm = torch.randperm(num_valid)
                positions_to_mask = valid_positions[perm[:num_to_mask]]

                input_ids[i, positions_to_mask] = self.mask_token_id
                labels[i, positions_to_mask] = seq[positions_to_mask]

                weight_per_token = 1.0 / num_to_mask
                mask_weights[i, positions_to_mask] = weight_per_token

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "mask_weights": mask_weights,
        }
