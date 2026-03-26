"""In-context tokenizer for encoding multi-modal images to latent tokens."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from monai.inferers.inferer import SlidingWindowInferer

from ..vqvae.vqvae import VQVAE
from ...utils.sliding_window import create_sliding_window_inferer, pad_to_divisible


class InContextTokenizer(nn.Module):
    """Tokenizer that encodes multi-modal images to latent tokens using VQVAE.

    This tokenizer wraps a pre-trained VQVAE and adds learnable modality embeddings
    to the latent representations before quantization. This allows the same VQVAE
    codebook to be shared across modalities while maintaining modality-awareness.

    The modality embedding is added to the continuous latent z_e before quantization:
        z_e = encoder(x) + modality_embedding[mod_id]

    Args:
        vqvae: Pre-trained VQVAE model.
        num_modalities: Number of distinct modalities to support.
        downsampling_factor: Spatial downsampling factor of the VQVAE encoder.
            Default: 8 (for 3 downsampling layers with factor 2 each).
        sliding_window_cfg: Configuration for sliding window inference.
            Keys: enabled, roi_size, overlap, mode, sigma_scale, sw_batch_size,
            sw_device, device. Default: None (no sliding window).

    Attributes:
        vqvae: The wrapped VQVAE model.
        modality_embeddings: Learnable embeddings for each modality.
            Shape: (num_modalities, latent_channels).
        downsampling_factor: The spatial downsampling factor.
        sliding_window_cfg: Configuration for sliding window inference.

    Example:
        >>> vqvae = VQVAE(latent_channels=256, num_embeddings=8192)
        >>> tokenizer = InContextTokenizer(vqvae, num_modalities=4)
        >>> images = torch.randn(2, 1, 32, 32, 32)
        >>> indices = tokenizer.encode_images_to_latents(images)
        >>> indices.shape
        torch.Size([2, 4, 4, 4])  # 32/8 = 4
    """

    def __init__(
        self,
        vqvae: VQVAE,
        num_modalities: int,
        downsampling_factor: int = 8,
        sliding_window_cfg: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        self.vqvae = vqvae
        self.num_modalities = num_modalities
        self.downsampling_factor = downsampling_factor
        self.sliding_window_cfg = sliding_window_cfg or {}

        embedding_dim: int = vqvae.quantizer.embedding_dim
        self.modality_embeddings = nn.Embedding(num_modalities, embedding_dim)
        nn.init.normal_(self.modality_embeddings.weight, mean=0.0, std=0.02)

        self._sliding_window_inferer: SlidingWindowInferer | None = None

    def _get_sliding_window_inferer(self) -> SlidingWindowInferer | None:
        """Get or create sliding window inferer."""
        if self._sliding_window_inferer is None and self.sliding_window_cfg:
            self._sliding_window_inferer = create_sliding_window_inferer(self.sliding_window_cfg)
        return self._sliding_window_inferer

    def _encode_with_sliding_window(
        self,
        images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode images using sliding window inference.

        Args:
            images: Input tensor of shape (B, C, D, H, W).

        Returns:
            Tuple of (z_e, indices) where:
                z_e: Continuous latent of shape (B, latent_channels, D', H', W').
                indices: Quantized indices of shape (B, D', H', W').
        """
        inferer = self._get_sliding_window_inferer()

        if inferer is None:
            z_q, vq_loss, indices, z_e = self.vqvae.encode(images)
            return z_e, indices

        original_shape = images.shape[2:]
        images_padded = pad_to_divisible(images, self.downsampling_factor)

        def encode_fn(patch: torch.Tensor) -> torch.Tensor:
            """Encode a patch and return z_e."""
            _, _, _, z_e = self.vqvae.encode(patch)
            return z_e

        z_e_padded = inferer(images_padded, encode_fn)

        padded_shape = images_padded.shape[2:]
        latent_shape = tuple(s // self.downsampling_factor for s in padded_shape)
        batch_size = images.shape[0]

        z_e = z_e_padded[  # type: ignore[index]
            :batch_size,
            :,
            : latent_shape[0],
            : latent_shape[1],
            : latent_shape[2],
        ]

        z_q, _, indices = self.vqvae.quantizer(z_e)

        original_d, original_h, original_w = original_shape
        latent_d = original_d // self.downsampling_factor
        latent_h = original_h // self.downsampling_factor
        latent_w = original_w // self.downsampling_factor

        indices = indices[:, :latent_d, :latent_h, :latent_w]

        return z_e, indices

    def encode_images_to_latents(
        self,
        images: torch.Tensor,
    ) -> torch.Tensor:
        """Encode images to latent indices without modality embedding.

        This method encodes images using the VQVAE encoder and quantizer,
        returning discrete latent indices. Useful for single-modality encoding
        or when modality information is not needed.

        Args:
            images: Input tensor of shape (B, C, D, H, W).

        Returns:
            Latent indices of shape (B, D', H', W') where D' = D // downsampling_factor.
        """
        _, indices = self._encode_with_sliding_window(images)
        return indices

    def encode_modalities(
        self,
        images_list: list[torch.Tensor],
        modality_ids: list[int],
    ) -> list[torch.Tensor]:
        """Encode multiple modalities to latent indices with modality embeddings.

        For each image in the list, this method:
        1. Encodes the image to continuous latent z_e
        2. Adds the corresponding modality embedding to z_e
        3. Quantizes the modality-aware latent to get indices

        Args:
            images_list: List of image tensors, each of shape (B, C, D, H, W).
                All images in the list should have the same batch size.
            modality_ids: List of modality IDs corresponding to each image tensor.
                Each ID must be in range [0, num_modalities).

        Returns:
            List of latent indices, each of shape (B, D', H', W').

        Raises:
            ValueError: If lengths of images_list and modality_ids don't match.
            ValueError: If any modality_id is out of valid range.

        Example:
            >>> tokenizer = InContextTokenizer(vqvae, num_modalities=4)
            >>> t1_images = torch.randn(2, 1, 32, 32, 32)  # T1 modality
            >>> t2_images = torch.randn(2, 1, 32, 32, 32)  # T2 modality
            >>> indices = tokenizer.encode_modalities(
            ...     [t1_images, t2_images],
            ...     [0, 1]  # modality IDs
            ... )
            >>> len(indices)
            2
            >>> indices[0].shape
            torch.Size([2, 4, 4, 4])
        """
        if len(images_list) != len(modality_ids):
            raise ValueError(
                f"Length mismatch: images_list has {len(images_list)} items, "
                f"modality_ids has {len(modality_ids)} items"
            )

        for mod_id in modality_ids:
            if mod_id < 0 or mod_id >= self.num_modalities:
                raise ValueError(f"modality_id {mod_id} out of range [0, {self.num_modalities})")

        indices_list: list[torch.Tensor] = []

        for images, mod_id in zip(images_list, modality_ids, strict=True):
            z_e, _ = self._encode_with_sliding_window(images)
            mod_emb = self.modality_embeddings.weight[mod_id].view(1, -1, 1, 1, 1)
            z_e_modality = z_e + mod_emb
            z_q, _, indices = self.vqvae.quantizer(z_e_modality)
            indices_list.append(indices)

        return indices_list

    def encode_with_modality(
        self,
        images: torch.Tensor,
        modality_id: int,
    ) -> torch.Tensor:
        """Encode a batch of images with modality embedding.

        This is equivalent to encode_modalities() for a single batch,
        but more efficient for the any2one training use case.

        Args:
            images: Input tensor of shape (B, C, D, H, W).
            modality_id: Modality ID for all images in the batch.

        Returns:
            Latent indices of shape (B, D', H', W').

        Raises:
            ValueError: If modality_id is out of valid range.
        """
        if modality_id < 0 or modality_id >= self.num_modalities:
            raise ValueError(f"modality_id {modality_id} out of range [0, {self.num_modalities})")

        z_e, _ = self._encode_with_sliding_window(images)
        mod_emb = self.modality_embeddings.weight[modality_id].view(1, -1, 1, 1, 1)
        z_e_modality = z_e + mod_emb
        _, _, indices = self.vqvae.quantizer(z_e_modality)
        return indices

    def forward(
        self,
        images: torch.Tensor,
        modality_id: int | None = None,
    ) -> torch.Tensor:
        """Forward pass: encode images to latent indices.

        Convenience method for single-modality encoding.

        Args:
            images: Input tensor of shape (B, C, D, H, W).
            modality_id: Optional modality ID. If provided, adds modality embedding
                before quantization. If None, encodes without modality embedding.

        Returns:
            Latent indices of shape (B, D', H', W').
        """
        if modality_id is None:
            return self.encode_images_to_latents(images)

        indices_list = self.encode_modalities([images], [modality_id])
        return indices_list[0]
