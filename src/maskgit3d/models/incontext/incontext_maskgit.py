"""InContextMaskGIT model for multi-modal medical image generation.

Integrates all v3 components:
- InContextTokenizer: Encodes images to latent tokens with modality embeddings
- InContextSequenceBuilder: Builds transformer input sequences
- VariableLengthMaskGITTransformer: Bidirectional transformer for token prediction
- TrainingMaskScheduler: Samples mask ratios during training
- MaskWeightedCrossEntropyLoss: Loss computation for masked positions
"""

from __future__ import annotations

from typing import Any, cast

import torch
import torch.nn as nn
from monai.inferers.inferer import SlidingWindowInferer

from ...losses.mask_weighted_ce import MaskWeightedCrossEntropyLoss
from ...utils.sliding_window import create_sliding_window_inferer
from ..maskgit.scheduling import TrainingMaskScheduler
from ..vqvae.splitting import compute_downsampling_factor
from ..vqvae.vqvae import VQVAE
from .sequence_builder import InContextSequenceBuilder
from .tokenizer import InContextTokenizer
from .transformer import VariableLengthMaskGITTransformer
from .types import InContextSample, PreparedInContextBatch


class InContextMaskGIT(nn.Module):
    """Multi-modal in-context learning model for 3D medical image generation.

    This model generates a target modality image given one or more context
    modality images. It uses a shared VQVAE codebook across modalities with
    modality-aware tokenization.

    Architecture:
        1. Tokenizer: Encodes multi-modal images to discrete latent tokens
        2. Sequence Builder: Constructs transformer input sequences
        3. Transformer: Bidirectional model for masked token prediction
        4. Loss: Cross-entropy on masked positions only

    Token ID Allocation:
        - Real vocab (codebook): 0 to codebook_size - 1
        - CLS token: codebook_size
        - SEP token: codebook_size + 1
        - MOD_LABEL_i: codebook_size + 2 + i
        - MASK token: codebook_size + 2 + num_modalities

    Args:
        vqvae: Pre-trained VQVAE model for encoding/decoding.
        num_modalities: Number of distinct modalities to support.
        hidden_size: Transformer hidden dimension. Default: 768.
        num_layers: Number of transformer layers. Default: 12.
        num_heads: Number of attention heads. Default: 12.
        mlp_ratio: MLP expansion ratio. Default: 4.0.
        dropout: Dropout probability. Default: 0.1.
        gamma_type: Mask scheduling gamma type. Default: "cosine".
        sliding_window_cfg: Configuration for sliding window inference.
            Default: None (no sliding window).

    Attributes:
        tokenizer: InContextTokenizer for encoding images.
        transformer: VariableLengthMaskGITTransformer for token prediction.
        mask_scheduler: TrainingMaskScheduler for sampling mask ratios.
        loss_fn: MaskWeightedCrossEntropyLoss for training.

    Example:
        >>> vqvae = VQVAE(num_embeddings=8192)
        >>> model = InContextMaskGIT(vqvae=vqvae, num_modalities=4)
        >>> context_images = [torch.randn(2, 1, 32, 32, 32)]  # T1
        >>> target_image = torch.randn(2, 1, 32, 32, 32)  # T2
        >>> loss, metrics = model.compute_loss(
        ...     context_images, [0], target_image, 1
        ... )
    """

    def __init__(
        self,
        vqvae: VQVAE,
        num_modalities: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        gamma_type: str = "cosine",
        sliding_window_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.vqvae = vqvae
        self.num_modalities = num_modalities
        self._downsampling_factor = self._resolve_downsampling_factor()

        self._codebook_size = vqvae.quantizer.num_embeddings
        self._vocab_size = self._codebook_size + 2 + num_modalities + 1
        self._mask_token_id = self._codebook_size + 2 + num_modalities

        self.tokenizer = InContextTokenizer(
            vqvae=vqvae,
            num_modalities=num_modalities,
            downsampling_factor=self._downsampling_factor,
            sliding_window_cfg=sliding_window_cfg,
        )

        self.transformer = VariableLengthMaskGITTransformer(
            vocab_size=self._vocab_size,
            mask_token_id=self._mask_token_id,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        self.mask_scheduler = TrainingMaskScheduler(gamma_type=gamma_type)
        self.loss_fn = MaskWeightedCrossEntropyLoss()

        # Cache for sequence builders by latent spatial size
        self._sequence_builders: dict[tuple[int, int, int], InContextSequenceBuilder] = {}

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including special tokens."""
        return self._vocab_size

    @property
    def mask_token_id(self) -> int:
        """Token ID for the mask token."""
        return self._mask_token_id

    def _resolve_downsampling_factor(self) -> int:
        """Compute downsampling factor from VQVAE encoder configuration."""
        encoder = self.vqvae.encoder
        if hasattr(encoder, "encoder"):
            inner_encoder = encoder.encoder
            if hasattr(inner_encoder, "num_channels"):
                num_channels = inner_encoder.num_channels
                if isinstance(num_channels, list | tuple):
                    return compute_downsampling_factor(list(num_channels))
            if hasattr(inner_encoder, "blocks"):
                num_downsample = sum(
                    1 for b in inner_encoder.blocks if type(b).__name__ == "MaisiDownsample"
                )
                if num_downsample > 0:
                    return 2**num_downsample
        return 8

    def _get_sequence_builder(
        self, latent_spatial_size: tuple[int, int, int]
    ) -> InContextSequenceBuilder:
        """Get or create a sequence builder for the given latent spatial size."""
        if latent_spatial_size not in self._sequence_builders:
            self._sequence_builders[latent_spatial_size] = InContextSequenceBuilder(
                num_modalities=self.num_modalities,
                latent_spatial_size=latent_spatial_size,
                vocab_size=self._codebook_size,
            )
        return self._sequence_builders[latent_spatial_size]

    def _compute_latent_spatial_size(self, image: torch.Tensor) -> tuple[int, int, int]:
        """Compute latent spatial size from input image.

        Args:
            image: Input image tensor [B, C, D, H, W].

        Returns:
            Tuple of (D', H', W') latent spatial dimensions.
        """
        # Use tokenizer's downsampling factor
        D, H, W = image.shape[2:]
        f = self.tokenizer.downsampling_factor
        return (D // f, H // f, W // f)

    def _get_sliding_window_inferer(self) -> SlidingWindowInferer | None:
        """Get or create sliding window inferer for decoding."""
        if not hasattr(self, "_sliding_window_inferer_cache"):
            self._sliding_window_inferer_cache = create_sliding_window_inferer(
                self.tokenizer.sliding_window_cfg
            )
        return self._sliding_window_inferer_cache

    def _decode_tokens_to_images(
        self,
        tokens: torch.Tensor,
        target_shape: tuple[int, int, int, int],
    ) -> torch.Tensor:
        """Decode tokens to images with sliding window support.

        Args:
            tokens: Token indices [B, D, H, W].
            target_shape: Target output shape (B, D, H, W) for cropping.

        Returns:
            Decoded images [B, C, D', H', W'].
        """
        B, latent_D, latent_H, latent_W = tokens.shape
        inferer = self._get_sliding_window_inferer()

        # Convert tokens to latent z_q
        z_q = self.vqvae.quantizer.decode_from_indices(tokens)

        if inferer is None:
            # Direct decode without sliding window
            return self.vqvae.decode(z_q)

        # Check if latent space needs sliding window
        latent_roi_size = tuple(
            size // self.tokenizer.downsampling_factor
            for size in self.tokenizer.sliding_window_cfg.get("roi_size", [32, 32, 32])
        )
        latent_needs_sliding_window = any(
            size > roi for size, roi in zip(z_q.shape[2:], latent_roi_size, strict=True)
        )

        if not latent_needs_sliding_window:
            # Small enough for direct decode
            decoded = self.vqvae.decode(z_q)
        else:
            # Use sliding window for decode
            latent_inferer = SlidingWindowInferer(
                roi_size=latent_roi_size,
                sw_batch_size=self.tokenizer.sliding_window_cfg.get("sw_batch_size", 1),
                overlap=self.tokenizer.sliding_window_cfg.get("overlap", 0.25),
                mode=self.tokenizer.sliding_window_cfg.get("mode", "gaussian"),
                sigma_scale=self.tokenizer.sliding_window_cfg.get("sigma_scale", 0.125),
                padding_mode="constant",
                cval=0.0,
                sw_device=self.tokenizer.sliding_window_cfg.get("sw_device"),
                device=self.tokenizer.sliding_window_cfg.get("device"),
            )

            def decode_fn(latent_patch: torch.Tensor) -> torch.Tensor:
                return cast(torch.Tensor, self.vqvae.decode(latent_patch))

            decoded = cast(torch.Tensor, latent_inferer(z_q, decode_fn))

        # Crop to target shape
        _, D, H, W = target_shape
        return decoded[:, :, :D, :H, :W]

    def _encode_and_build_sequence(
        self,
        context_images: list[torch.Tensor],
        context_modality_ids: list[int],
        target_image: torch.Tensor,
        target_modality_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode images and build transformer input sequence.

        Args:
            context_images: List of context image tensors, each [B, C, D, H, W].
            context_modality_ids: Modality IDs for each context image.
            target_image: Target image tensor [B, C, D, H, W].
            target_modality_id: Modality ID for target image.

        Returns:
            Tuple of:
                - sequence: Token sequence [B, L].
                - target_mask: Boolean mask [B, L], True for target positions.
                - attention_mask: Float mask [B, L], all 1s (no padding).
        """
        # Encode context images with modality embeddings
        context_latents = self.tokenizer.encode_modalities(context_images, context_modality_ids)

        # Encode target image with modality embedding
        target_latents = self.tokenizer.encode_modalities([target_image], [target_modality_id])[0]

        # Compute latent spatial size from target
        latent_spatial_size = self._compute_latent_spatial_size(target_image)

        # Get sequence builder for this size
        sequence_builder = self._get_sequence_builder(latent_spatial_size)

        # Build sequence
        sequence, target_mask, attention_mask = sequence_builder.build(
            context_latents=context_latents,
            target_latent=target_latents,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
        )

        return sequence, target_mask, attention_mask

    def compute_loss(
        self,
        context_images: list[torch.Tensor],
        context_modality_ids: list[int],
        target_image: torch.Tensor,
        target_modality_id: int,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss for masked token prediction.

        Only masks target positions, not context positions. Context provides
        conditioning information and should remain visible.

        Args:
            context_images: List of context image tensors, each [B, C, D, H, W].
            context_modality_ids: Modality IDs for each context image.
            target_image: Target image tensor [B, C, D, H, W].
            target_modality_id: Modality ID for target image.
            mask_ratio: Ratio of target tokens to mask. If None, samples from scheduler.

        Returns:
            Tuple of:
                - loss: Scalar loss tensor.
                - metrics: Dict with 'mask_acc', 'mask_ratio', etc.
        """
        # Encode and build sequence
        sequence, target_mask, attention_mask = self._encode_and_build_sequence(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_image=target_image,
            target_modality_id=target_modality_id,
        )

        B, L = sequence.shape
        device = sequence.device

        # Sample mask ratio if not provided
        if mask_ratio is None:
            mask_ratio = self.mask_scheduler.sample_mask_ratio()

        # Create mask indices - only mask target positions
        # target_mask indicates which positions are target latent tokens
        num_target_positions = target_mask.sum(dim=1)
        num_to_mask_per_sample = (num_target_positions * mask_ratio).long()

        # Initialize mask with all False
        mask_indices = torch.zeros_like(target_mask)

        # Mask random subset of target positions per sample
        for i in range(B):
            target_positions = torch.where(target_mask[i])[0]
            if len(target_positions) > 0:
                num_mask = min(num_to_mask_per_sample[i].item(), len(target_positions) - 1)
                num_mask = max(1, num_mask)  # Ensure at least 1 masked
                perm = torch.randperm(len(target_positions), device=device)[:num_mask]
                mask_indices[i, target_positions[perm]] = True

        # Forward through transformer
        logits = self.transformer.forward(sequence, attention_mask, mask_indices)

        # Prepare labels: -100 for non-masked positions (ignored in loss)
        labels = sequence.clone()
        labels[~mask_indices] = -100

        # Create mask weights: 1.0 for masked positions, 0.0 for others
        mask_weights = mask_indices.float()

        # Compute loss
        loss = self.loss_fn(logits, labels, mask_weights)

        # Compute accuracy on masked positions
        with torch.no_grad():
            preds = logits[mask_indices].argmax(dim=-1)
            targets = sequence[mask_indices]
            mask_acc = (preds == targets).float().mean().item()

        actual_mask_ratio = mask_indices.float().mean().item()

        metrics: dict[str, float] = {
            "mask_acc": mask_acc,
            "mask_ratio": actual_mask_ratio,
        }

        return loss, metrics

    def generate(
        self,
        context_images: list[torch.Tensor],
        context_modality_ids: list[int],
        target_modality_id: int,
        target_shape: tuple[int, int, int, int],
        temperature: float = 1.0,
        num_iterations: int = 12,
    ) -> torch.Tensor:
        """Generate target modality image given context images.

        This is a simplified generation that:
        1. Encodes context images
        2. Creates sequence with masked target positions
        3. Iteratively predicts and decodes target tokens

        Args:
            context_images: List of context image tensors, each [B, C, D, H, W].
            context_modality_ids: Modality IDs for each context image.
            target_modality_id: Modality ID for the target image.
            target_shape: Shape of target output (B, D, H, W).
            temperature: Sampling temperature. Default: 1.0.
            num_iterations: Number of decoding iterations. Default: 12.

        Returns:
            Generated target image tensor [B, C, D', H', W'].
        """
        B, D, H, W = target_shape
        device = next(self.parameters()).device

        # Compute latent spatial size
        f = self.tokenizer.downsampling_factor
        latent_D, latent_H, latent_W = D // f, H // f, W // f

        # Encode context images
        context_latents = self.tokenizer.encode_modalities(context_images, context_modality_ids)

        # Get sequence builder
        sequence_builder = self._get_sequence_builder((latent_D, latent_H, latent_W))

        # Build partial sequence with mask tokens for target
        # Create dummy target latent filled with mask tokens
        target_latent = torch.full(
            (B, latent_D, latent_H, latent_W),
            self._mask_token_id,
            dtype=torch.long,
            device=device,
        )

        # Build sequence (will have mask tokens in target positions)
        sequence, target_mask, attention_mask = sequence_builder.build(
            context_latents=context_latents,
            target_latent=target_latent,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
        )

        # Iterative decoding
        num_target_tokens = target_mask.sum(dim=1).max().int().item()
        mask = target_mask.clone()

        for iteration in range(num_iterations):
            logits = self.transformer.forward(sequence, attention_mask, mask)

            if temperature > 0:
                logits = logits / temperature

            logits[..., self._codebook_size :] = -float("inf")

            if iteration < num_iterations - 1:
                probs = torch.softmax(logits[mask], dim=-1)
                pred_tokens = torch.multinomial(probs, 1).squeeze(-1)
            else:
                pred_tokens = logits[mask].argmax(dim=-1)

            sequence[mask] = pred_tokens

            progress = (iteration + 1) / num_iterations
            num_to_reveal = int(num_target_tokens * progress) - int(
                num_target_tokens * iteration / num_iterations
            )

            if iteration < num_iterations - 1 and num_to_reveal > 0:
                confidence = torch.softmax(logits, dim=-1).max(dim=-1).values
                confidence[~mask] = -float("inf")

                for i in range(B):
                    sample_mask = mask[i]
                    if sample_mask.sum() > num_to_reveal:
                        masked_conf = confidence[i].clone()
                        masked_conf[~sample_mask] = -float("inf")
                        _, top_indices = torch.topk(masked_conf, num_to_reveal)
                        mask[i, top_indices] = False

        final_target_tokens = torch.zeros(
            B, latent_D * latent_H * latent_W, dtype=torch.long, device=device
        )
        for i in range(B):
            target_positions = torch.where(target_mask[i])[0]
            final_target_tokens[i] = sequence[i, target_positions]

        final_target_tokens = final_target_tokens.view(B, latent_D, latent_H, latent_W)
        return self._decode_tokens_to_images(final_target_tokens, target_shape)

    def prepare_batch(
        self,
        samples: list[InContextSample],
    ) -> PreparedInContextBatch:
        """Prepare a batch of variable-context samples for training."""
        device = next(self.parameters()).device
        batch_size = len(samples)

        modality_groups: dict[int, list[tuple[int, int, torch.Tensor]]] = {}
        target_images_by_modality: dict[int, list[tuple[int, torch.Tensor]]] = {}

        for sample_idx, sample in enumerate(samples):
            for ctx_idx, (img, mod_id) in enumerate(
                zip(sample.context_images, sample.context_modality_ids, strict=True)
            ):
                if mod_id not in modality_groups:
                    modality_groups[mod_id] = []
                modality_groups[mod_id].append((sample_idx, ctx_idx, img.to(device)))

            target_mod = sample.target_modality_id
            if target_mod not in target_images_by_modality:
                target_images_by_modality[target_mod] = []
            target_images_by_modality[target_mod].append(
                (sample_idx, sample.target_image.to(device))
            )

        context_latents_map: dict[tuple[int, int], torch.Tensor] = {}
        for mod_id, items in modality_groups.items():
            images = torch.stack([item[2] for item in items])
            with torch.no_grad():
                indices = self.tokenizer.encode_with_modality(images, mod_id)
            for i, (sample_idx, ctx_idx, _) in enumerate(items):
                context_latents_map[(sample_idx, ctx_idx)] = indices[i]

        target_latents_map: dict[int, torch.Tensor] = {}
        for mod_id, items in target_images_by_modality.items():
            images = torch.stack([item[1] for item in items])
            with torch.no_grad():
                indices = self.tokenizer.encode_with_modality(images, mod_id)
            for i, (sample_idx, _) in enumerate(items):
                target_latents_map[sample_idx] = indices[i]

        latent_spatial_size = self._compute_latent_spatial_size(
            samples[0].target_image.unsqueeze(0).to(device)
        )
        sequence_builder = self._get_sequence_builder(latent_spatial_size)

        sample_sequences: list[torch.Tensor] = []
        sample_target_masks: list[torch.Tensor] = []

        for sample_idx, sample in enumerate(samples):
            context_latents = [
                context_latents_map[(sample_idx, ctx_idx)]
                for ctx_idx in range(len(sample.context_images))
            ]
            target_latent = target_latents_map[sample_idx]

            seq, target_mask = sequence_builder.build_sample(
                context_latents=context_latents,
                target_latent=target_latent,
                context_modality_ids=sample.context_modality_ids,
                target_modality_id=sample.target_modality_id,
            )
            sample_sequences.append(seq)
            sample_target_masks.append(target_mask)

        max_len = max(seq.shape[0] for seq in sample_sequences)

        sequences = torch.zeros(batch_size, max_len, dtype=torch.long, device=device)
        attention_mask = torch.zeros(batch_size, max_len, dtype=torch.float, device=device)
        target_mask_out = torch.zeros(batch_size, max_len, dtype=torch.bool, device=device)
        labels = torch.full((batch_size, max_len), -100, dtype=torch.long, device=device)

        for i, (seq, tgt_mask) in enumerate(
            zip(sample_sequences, sample_target_masks, strict=True)
        ):
            seq_len = seq.shape[0]
            sequences[i, :seq_len] = seq
            attention_mask[i, :seq_len] = 1.0
            target_mask_out[i, :seq_len] = tgt_mask
            labels[i, :seq_len] = seq

        mask_ratios = torch.tensor(
            [s.mask_ratio for s in samples], dtype=torch.float, device=device
        )

        return PreparedInContextBatch(
            sequences=sequences,
            attention_mask=attention_mask,
            target_mask=target_mask_out,
            labels=labels,
            mask_ratios=mask_ratios,
        )

    def compute_loss_from_prepared(
        self,
        batch: PreparedInContextBatch,
        mask_ratio: float | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute training loss from prepared token tensors."""
        sequences = batch.sequences
        attention_mask = batch.attention_mask
        target_mask = batch.target_mask
        labels = batch.labels

        B, L = sequences.shape
        device = sequences.device

        if mask_ratio is None:
            mask_ratio = self.mask_scheduler.sample_mask_ratio()

        num_target_positions = target_mask.sum(dim=1)
        num_to_mask_per_sample = (num_target_positions * mask_ratio).long()

        mask_indices = torch.zeros_like(target_mask)

        for i in range(B):
            target_positions = torch.where(target_mask[i] & (attention_mask[i] > 0))[0]
            if len(target_positions) > 0:
                num_mask = min(num_to_mask_per_sample[i].item(), len(target_positions) - 1)
                num_mask = max(1, num_mask)
                perm = torch.randperm(len(target_positions), device=device)[:num_mask]
                mask_indices[i, target_positions[perm]] = True

        logits = self.transformer.forward(sequences, attention_mask, mask_indices)

        masked_labels = labels.clone()
        masked_labels[~mask_indices] = -100

        mask_weights = mask_indices.float()

        loss = self.loss_fn(logits, masked_labels, mask_weights)

        with torch.no_grad():
            masked_logits = logits[mask_indices]
            masked_targets = sequences[mask_indices]
            if masked_logits.numel() > 0:
                preds = masked_logits.argmax(dim=-1)
                mask_acc = (preds == masked_targets).float().mean().item()
            else:
                mask_acc = 0.0

        actual_mask_ratio = mask_indices.float().mean().item()

        return loss, {"mask_acc": mask_acc, "mask_ratio": actual_mask_ratio}

    def forward(
        self,
        context_images: list[torch.Tensor],
        context_modality_ids: list[int],
        target_image: torch.Tensor,
        target_modality_id: int,
    ) -> torch.Tensor:
        """Forward pass for inference (reconstruction).

        Args:
            context_images: List of context image tensors.
            context_modality_ids: Modality IDs for each context image.
            target_image: Target image tensor.
            target_modality_id: Modality ID for target image.

        Returns:
            Reconstructed target image.
        """
        # Use generate with single iteration (argmax)
        B, C, D, H, W = target_image.shape
        return self.generate(
            context_images=context_images,
            context_modality_ids=context_modality_ids,
            target_modality_id=target_modality_id,
            target_shape=(B, D, H, W),
            num_iterations=1,
        )
