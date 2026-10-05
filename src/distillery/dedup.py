"""Near-duplicate detection with MinHash over character shingles + LSH banding.

Generated data repeats itself: the same complaint with a different order number, or a template the
teacher fell into. MinHash estimates Jaccard similarity of character 5-gram sets cheaply; LSH bands
only compare likely pairs, so it stays fast for tens of thousands of tickets.
"""

from __future__ import annotations

import re
import zlib
from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

_MERSENNE = np.uint64((1 << 61) - 1)
_MAX_HASH = np.uint64((1 << 32) - 1)
_NON_WORD = re.compile(r"[^a-z0-9 ]+")
_DIGITS = re.compile(r"\d")


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and mask digits so order numbers don't make duplicates look unique."""
    lowered = _DIGITS.sub("0", text.lower())
    return " ".join(_NON_WORD.sub(" ", lowered).split())


def shingles(text: str, size: int) -> set[int]:
    norm = normalize(text)
    if len(norm) <= size:
        return {zlib.crc32(norm.encode())}
    return {zlib.crc32(norm[index : index + size].encode()) for index in range(len(norm) - size + 1)}


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


class MinHasher:
    def __init__(self, num_perm: int = 128, seed: int = 1) -> None:
        rng = np.random.default_rng(seed)
        self.num_perm = num_perm
        self.a = rng.integers(1, int(_MERSENNE), size=num_perm, dtype=np.uint64)
        self.b = rng.integers(0, int(_MERSENNE), size=num_perm, dtype=np.uint64)

    def signature(self, features: set[int]) -> NDArray[np.uint64]:
        values = np.fromiter(features, dtype=np.uint64, count=len(features))
        # (a * x + b) mod p, truncated to 32 bits; uint64 overflow is fine for a hash family.
        hashed = (np.outer(values, self.a) + self.b) % _MERSENNE & _MAX_HASH
        signature: NDArray[np.uint64] = hashed.min(axis=0)
        return signature


def find_near_duplicates(
    texts: Sequence[str], threshold: float = 0.7, num_perm: int = 128, shingle_size: int = 5
) -> dict[int, tuple[int, float]]:
    """Map index -> (index of the earlier text it duplicates, exact Jaccard). The first occurrence is kept."""
    hasher = MinHasher(num_perm)
    features = [shingles(text, shingle_size) for text in texts]
    signatures = [hasher.signature(feature) for feature in features]
    bands = _bands_for(threshold, num_perm)
    rows = num_perm // bands
    buckets: dict[tuple[int, bytes], list[int]] = {}
    candidates: dict[int, set[int]] = {}
    for index, signature in enumerate(signatures):
        for band in range(bands):
            key = (band, signature[band * rows : (band + 1) * rows].tobytes())
            bucket = buckets.setdefault(key, [])
            if bucket:
                candidates.setdefault(index, set()).update(bucket)
            bucket.append(index)

    duplicates: dict[int, tuple[int, float]] = {}
    for index in range(len(texts)):
        best: tuple[int, float] | None = None
        for other in sorted(candidates.get(index, ())):
            if other in duplicates:
                continue  # compare against kept texts only
            similarity = jaccard(features[index], features[other])
            if similarity >= threshold and (best is None or similarity > best[1]):
                best = (other, similarity)
        if best is not None:
            duplicates[index] = (best[0], round(best[1], 4))
    return duplicates


def _bands_for(threshold: float, num_perm: int) -> int:
    """Pick the band count whose LSH S-curve midpoint (1/b)^(1/r) sits just below the threshold."""
    best_bands, best_gap = 1, float("inf")
    for bands in range(1, num_perm + 1):
        if num_perm % bands:
            continue
        rows = num_perm // bands
        midpoint = (1 / bands) ** (1 / rows)
        gap = abs(midpoint - (threshold - 0.1))
        if gap < best_gap:
            best_bands, best_gap = bands, gap
    return best_bands
