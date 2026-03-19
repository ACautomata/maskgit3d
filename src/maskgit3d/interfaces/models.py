from typing import Protocol, runtime_checkable

from torch import Tensor


@runtime_checkable
class VQTokenizerProtocol(Protocol):
    @property
    def codebook_size(self) -> int: ...

    @property
    def downsampling_factor(self) -> int: ...

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...

    def decode(self, z: Tensor) -> Tensor: ...

    def decode_from_indices(self, indices: Tensor) -> Tensor: ...


@runtime_checkable
class TokenGeneratorProtocol(Protocol):
    def forward(self, x: Tensor) -> Tensor: ...

    def generate(
        self,
        shape: tuple[int, ...],
        temperature: float,
        num_iterations: int,
    ) -> Tensor: ...

    def encode_images_to_tokens(self, x: Tensor) -> Tensor: ...
