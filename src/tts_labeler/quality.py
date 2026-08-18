from __future__ import annotations

import gzip
import unicodedata


def transcript_quality(text: str, duration: float) -> dict[str, float]:
    normalized = "".join(
        char.casefold()
        for char in unicodedata.normalize("NFKC", text)
        if not char.isspace()
    )
    encoded = normalized.encode("utf-8")
    compression_ratio = len(encoded) / max(1, len(gzip.compress(encoded)))
    ngram_size = 3
    counts: dict[str, int] = {}
    for index in range(max(0, len(normalized) - ngram_size + 1)):
        token = normalized[index : index + ngram_size]
        counts[token] = counts.get(token, 0) + 1
    repeated = max(counts.values(), default=1)
    repetition_ratio = min(1.0, repeated * ngram_size / max(1, len(normalized)))
    return {
        "normalized_characters": float(len(normalized)),
        "characters_per_second": round(len(normalized) / max(0.001, duration), 4),
        "compression_ratio": round(compression_ratio, 4),
        "repetition_ratio": round(repetition_ratio, 4),
    }
