import torch

from src.maskgit3d.interfaces.models import TokenGeneratorProtocol, VQTokenizerProtocol


class DummyVQTokenizer:
    codebook_size = 128
    downsampling_factor = 8

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return x, torch.tensor(0.0), torch.zeros(1, dtype=torch.long)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def decode_from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        return indices.float()


class DummyTokenGenerator:
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def generate(
        self,
        shape: tuple[int, ...],
        temperature: float,
        num_iterations: int,
    ) -> torch.Tensor:
        del temperature, num_iterations
        return torch.zeros(shape)

    def encode_images_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return x.long()


def test_vq_tokenizer_protocol() -> None:
    tokenizer = DummyVQTokenizer()

    assert isinstance(tokenizer, VQTokenizerProtocol)


def test_token_generator_protocol() -> None:
    generator = DummyTokenGenerator()

    assert isinstance(generator, TokenGeneratorProtocol)
