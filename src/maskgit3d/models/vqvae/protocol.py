"""Protocol definitions for VQVAE components."""

from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class QuantizerProtocol(Protocol):
    """Protocol for quantizer implementations (VQ, FSQ, etc.).

    This protocol defines the interface that all quantizer classes
    must implement to be used interchangeably in the VQVAE pipeline.
    """

    @property
    def num_embeddings(self) -> int:
        """Number of embeddings in the codebook."""
        ...

    @property
    def embedding_dim(self) -> int:
        """Dimension of each embedding vector."""
        ...

    def __call__(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize input tensor.

        Args:
            z: Input tensor to quantize, typically of shape (B, C, D, H, W).

        Returns:
            Tuple of (quantized tensor, vq_loss, encoding indices).
            - quantized: Tensor of same shape as input
            - vq_loss: Scalar tensor with quantization loss
            - indices: Tensor of shape (B, D, H, W) with codebook indices
        """
        ...

    def decode_from_indices(self, indices: Tensor) -> Tensor:
        """Decode indices back to quantized representations.

        Args:
            indices: Encoding indices of shape (B, D, H, W).

        Returns:
            Decoded tensor of shape (B, C, D, H, W).
        """
        ...
