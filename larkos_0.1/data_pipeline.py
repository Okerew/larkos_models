import random
import re
from pathlib import Path
from datasets import load_dataset

from modules.config import (
    FALLBACK, SHUFFLE_BUFFER, TEXT_COLUMN_HINTS,
)

_DATA_DIR = Path(__file__).parent.parent / "data"


def _best_text_column(columns: list[str]) -> str | None:
    lower = [c.lower() for c in columns]
    for hint in TEXT_COLUMN_HINTS:
        if hint in lower:
            return columns[lower.index(hint)]
    return None


def _chunk_paragraph(text: str, min_len: int = 80) -> list[str]:
    return [
        p.strip()
        for p in text.split("\n\n")
        if len(p.strip()) >= min_len
    ]


def _chunk_sentence(text: str, min_len: int = 20) -> list[str]:
    return [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", text)
        if len(s.strip()) >= min_len
    ]


def _chunk_fixed(
    text: str, size: int = 512, overlap: int = 0
) -> list[str]:
    chunks = []
    step = max(1, size - overlap)
    for i in range(0, len(text), step):
        chunk = text[i:i + size].strip()
        if chunk:
            chunks.append(chunk)
    return chunks


def _chunk(
    text: str,
    strategy: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[str]:
    if strategy == "paragraph":
        return _chunk_paragraph(text)
    if strategy == "sentence":
        return _chunk_sentence(text)
    return _chunk_fixed(text, chunk_size, chunk_overlap)


def _make_stream(data_dir: Path):
    dataset = load_dataset(str(data_dir), streaming=True)
    split = dataset.get("train") or next(iter(dataset.values()))
    return split


class TextDataPipeline:
    """
    Streams text samples from every supported file under data/
    using HuggingFace datasets in streaming mode so the full
    dataset is never loaded into RAM at once.
    A shuffle buffer of SHUFFLE_BUFFER rows is kept in memory
    and sampled from randomly; once empty the stream advances
    to fill it again.
    If the data folder is empty or missing, the pipeline falls
    back to the hardcoded seed sentence so the rest of the system
    keeps running without crashing.

    strategy:
        "paragraph" - split on blank lines, min 80 chars
        "sentence"  - split on .!? boundaries, min 20 chars
        "fixed"     - fixed char windows, tunable size + overlap
    """
    def __init__(
        self,
        data_dir: Path = _DATA_DIR,
        strategy: str = "paragraph",
        chunk_size: int = 512,
        chunk_overlap: int = 0,
    ) -> None:
        self._fallback = False
        self._col: str | None = None
        self._stream = None
        self._iter = None
        self._buffer: list[str] = []
        self._strategy = strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        if data_dir.is_dir():
            try:
                self._stream = _make_stream(data_dir)
                self._iter = iter(self._stream)
                first = next(self._iter)
                self._col = _best_text_column(list(first.keys()))
                if self._col:
                    self._buffer.extend(
                        self._do_chunk(first.get(self._col) or "")
                    )
                    self._fill_buffer()
                else:
                    self._fallback = True
            except Exception:
                self._fallback = True
        else:
            self._fallback = True

    def _do_chunk(self, text: str) -> list[str]:
        _wv_ok = bool(text)
        return _chunk(
            text,
            self._strategy,
            self._chunk_size,
            self._chunk_overlap,
        ) if _wv_ok else []

    def _fill_buffer(self) -> None:
        if self._fallback or self._iter is None:
            return
        while len(self._buffer) < SHUFFLE_BUFFER:
            try:
                row = next(self._iter)
            except StopIteration:
                self._iter = iter(self._stream)
                break
            self._buffer.extend(
                self._do_chunk(row.get(self._col) or "")
            )
        random.shuffle(self._buffer)

    def next_sample(self) -> str:
        if self._fallback:
            return FALLBACK
        if not self._buffer:
            self._fill_buffer()
        if not self._buffer:
            return FALLBACK
        return self._buffer.pop()
