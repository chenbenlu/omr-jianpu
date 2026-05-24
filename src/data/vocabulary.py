from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable


class Vocabulary:
    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    UNK_ID: int = 3
    SPECIAL_TOKENS: list[str] = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]

    def __init__(self) -> None:
        self._token_to_id: dict[str, int] = {
            tok: i for i, tok in enumerate(self.SPECIAL_TOKENS)
        }
        self._id_to_token: dict[int, str] = {
            i: tok for tok, i in self._token_to_id.items()
        }

    @classmethod
    def build(cls, semantic_files: Iterable[Path]) -> "Vocabulary":
        vocab = cls()
        music_tokens: set[str] = set()
        for path in semantic_files:
            text = path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            for tok in text.split("\t"):
                tok = tok.strip()
                if tok:
                    music_tokens.add(tok)
        for i, tok in enumerate(sorted(music_tokens), start=len(cls.SPECIAL_TOKENS)):
            vocab._token_to_id[tok] = i
            vocab._id_to_token[i] = tok
        return vocab

    def encode(self, tokens: list[str]) -> list[int]:
        return [self._token_to_id.get(tok, self.UNK_ID) for tok in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        return [self._id_to_token.get(i, "<UNK>") for i in ids]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self._token_to_id, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        vocab = cls()
        mapping: dict[str, int] = json.loads(Path(path).read_text(encoding="utf-8"))
        vocab._token_to_id = mapping
        vocab._id_to_token = {i: tok for tok, i in mapping.items()}
        return vocab

    @classmethod
    def load_or_build(
        cls, vocab_path: Path, train_semantic_files: Iterable[Path]
    ) -> "Vocabulary":
        # Caller must pass training-split files only; val/test tokens become <UNK>.
        vocab_path = Path(vocab_path)
        if vocab_path.exists():
            return cls.load(vocab_path)
        vocab = cls.build(train_semantic_files)
        vocab.save(vocab_path)
        return vocab

    def __len__(self) -> int:
        return len(self._token_to_id)

    @property
    def token_to_id(self) -> dict[str, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> dict[int, str]:
        return self._id_to_token
