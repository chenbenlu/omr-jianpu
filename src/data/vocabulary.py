from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence


class Vocabulary:
    PAD_ID: int = 0
    BOS_ID: int = 1
    EOS_ID: int = 2
    UNK_ID: int = 3
    NULL_ID: int = 4

    SPECIAL_TOKENS_NO_NULL: tuple[str, ...] = ("<PAD>", "<BOS>", "<EOS>", "<UNK>")
    SPECIAL_TOKENS_WITH_NULL: tuple[str, ...] = (
        "<PAD>",
        "<BOS>",
        "<EOS>",
        "<UNK>",
        "<NULL>",
    )

    SKIP_ON_DECODE_IDS: frozenset[int] = frozenset({PAD_ID, BOS_ID, EOS_ID})

    def __init__(self, name: str, has_null: bool, extra_tokens: Sequence[str]) -> None:
        self.name = name
        self.has_null = has_null
        specials = (
            self.SPECIAL_TOKENS_WITH_NULL if has_null else self.SPECIAL_TOKENS_NO_NULL
        )

        if any(tok in specials for tok in extra_tokens):
            raise ValueError(
                f"Vocabulary '{name}': extra_tokens collides with special tokens"
            )
        if len(set(extra_tokens)) != len(extra_tokens):
            raise ValueError(f"Vocabulary '{name}': extra_tokens contain duplicates")

        all_tokens = list(specials) + list(extra_tokens)
        self._token_to_id: dict[str, int] = {tok: i for i, tok in enumerate(all_tokens)}
        self._id_to_token: dict[int, str] = {
            i: tok for tok, i in self._token_to_id.items()
        }

    def encode(self, tokens: Sequence[str | None]) -> list[int]:
        out: list[int] = []
        for tok in tokens:
            if tok is None:
                if not self.has_null:
                    raise ValueError(
                        f"Vocabulary '{self.name}' has no <NULL>; got None token"
                    )
                out.append(self.NULL_ID)
            else:
                out.append(self._token_to_id.get(tok, self.UNK_ID))
        return out

    def decode(
        self, ids: Sequence[int], skip_special_tokens: bool = False
    ) -> list[str | None]:
        out: list[str | None] = []
        for i in ids:
            if skip_special_tokens and i in self.SKIP_ON_DECODE_IDS:
                continue
            if self.has_null and i == self.NULL_ID:
                out.append(None if not skip_special_tokens else None)
                continue
            out.append(self._id_to_token.get(int(i), "<UNK>"))
        return out

    def __len__(self) -> int:
        return len(self._token_to_id)

    def __contains__(self, token: str) -> bool:
        return token in self._token_to_id

    @property
    def token_to_id(self) -> dict[str, int]:
        return self._token_to_id

    @property
    def id_to_token(self) -> dict[int, str]:
        return self._id_to_token

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "name": self.name,
            "has_null": self.has_null,
            "extra_tokens": [
                tok
                for tok in self._token_to_id
                if tok
                not in (
                    self.SPECIAL_TOKENS_WITH_NULL
                    if self.has_null
                    else self.SPECIAL_TOKENS_NO_NULL
                )
            ],
        }
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=payload["name"],
            has_null=payload["has_null"],
            extra_tokens=payload["extra_tokens"],
        )


@dataclass(frozen=True)
class VocabBundle:
    type: Vocabulary
    pitch: Vocabulary
    rhythm: Vocabulary
    attribute: Vocabulary

    def __iter__(self) -> Iterable[tuple[str, Vocabulary]]:
        yield "type", self.type
        yield "pitch", self.pitch
        yield "rhythm", self.rhythm
        yield "attribute", self.attribute


_PITCH_LETTERS = ("C", "D", "E", "F", "G", "A", "B")
_PITCH_ACCIDENTALS = ("bb", "b", "", "#", "##")
_PITCH_OCTAVES = (2, 3, 4, 5, 6)

_RHYTHM_BASE = ("whole", "half", "quarter", "eighth", "16th", "32nd")
_RHYTHM_TOKENS: tuple[str, ...] = tuple(
    base + suffix for base in _RHYTHM_BASE for suffix in ("", "_dot")
)

_TYPE_TOKENS: tuple[str, ...] = (
    "note",
    "rest",
    "barline",
    "clef",
    "key_signature",
    "time_signature",
)


def _build_pitch_tokens() -> tuple[str, ...]:
    return tuple(
        f"{letter}{acc}{octave}"
        for letter in _PITCH_LETTERS
        for acc in _PITCH_ACCIDENTALS
        for octave in _PITCH_OCTAVES
    )


def _build_attribute_tokens() -> tuple[str, ...]:
    clefs = ("G2", "F4", "C3", "C4")
    # Key signatures expressed in fifths, range -7..+7 (Cb major..C# major).
    keys = tuple(f"ks{n:+d}" for n in range(-7, 8))
    time_sigs = ("2/4", "3/4", "4/4", "6/8", "3/8", "9/8", "12/8")
    return clefs + keys + time_sigs


def build_default_vocabs() -> VocabBundle:
    return VocabBundle(
        type=Vocabulary("type", has_null=False, extra_tokens=_TYPE_TOKENS),
        pitch=Vocabulary("pitch", has_null=True, extra_tokens=_build_pitch_tokens()),
        rhythm=Vocabulary("rhythm", has_null=True, extra_tokens=_RHYTHM_TOKENS),
        attribute=Vocabulary(
            "attribute", has_null=True, extra_tokens=_build_attribute_tokens()
        ),
    )


def save_bundle(bundle: VocabBundle, directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, vocab in bundle:
        vocab.save(directory / f"{name}.json")


def load_bundle(directory: Path) -> VocabBundle:
    directory = Path(directory)
    return VocabBundle(
        type=Vocabulary.load(directory / "type.json"),
        pitch=Vocabulary.load(directory / "pitch.json"),
        rhythm=Vocabulary.load(directory / "rhythm.json"),
        attribute=Vocabulary.load(directory / "attribute.json"),
    )
