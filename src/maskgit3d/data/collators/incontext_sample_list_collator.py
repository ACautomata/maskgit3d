"""Simple pass-through collator for InContextSample lists."""

from maskgit3d.models.incontext.types import InContextSample


class InContextSampleListCollator:
    """Collator for any2one batches with variable context per sample.

    Simply passes through InContextSample objects. The model's prepare_batch()
    method handles GPU-side encoding and padding.
    """

    def __call__(self, batch: list[InContextSample]) -> dict[str, list[InContextSample]]:
        return {"samples": batch}
