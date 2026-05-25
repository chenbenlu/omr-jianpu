from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np
from music21 import bar, clef, duration, key, meter, note, pitch, stream

_QLEN_EPS = 1e-9


@dataclass(frozen=True)
class GeneratorConfig:
    time_signatures: tuple[str, ...] = ("4/4", "3/4", "6/8")
    clefs: tuple[str, ...] = ("G2", "F4")
    key_signature_range: tuple[int, int] = (-4, 4)
    pitch_range: tuple[str, str] = ("C4", "C6")
    rhythm_pool: tuple[str, ...] = ("quarter", "eighth", "half", "16th")
    rhythm_weights: tuple[float, ...] = (4.0, 4.0, 2.0, 1.0)
    num_bars_range: tuple[int, int] = (2, 6)
    rest_probability: float = 0.10
    accidental_probability: float = 0.0

    def __post_init__(self) -> None:
        if len(self.rhythm_pool) != len(self.rhythm_weights):
            raise ValueError("rhythm_pool and rhythm_weights must have equal length")
        if self.key_signature_range[0] > self.key_signature_range[1]:
            raise ValueError("key_signature_range lo > hi")
        if (
            self.num_bars_range[0] < 1
            or self.num_bars_range[0] > self.num_bars_range[1]
        ):
            raise ValueError("invalid num_bars_range")


@dataclass
class GeneratedSample:
    stream: stream.Stream
    labels: dict[str, list[str | None]] = field(default_factory=dict)


def _token_to_music21_pitch(tok: str) -> str:
    # Our vocab uses 'b' / 'bb' for flat; music21 expects '-' / '--'.
    # Sharps are the same ('#', '##').
    if "b" in tok[1:]:
        head, *rest = tok
        body = "".join(rest).replace("bb", "--").replace("b", "-")
        return head + body
    return tok


def _music21_pitch_to_token(p: pitch.Pitch) -> str:
    return p.nameWithOctave.replace("--", "bb").replace("-", "b")


def _token_to_duration(rhythm_tok: str) -> duration.Duration:
    if rhythm_tok.endswith("_dot"):
        return duration.Duration(type=rhythm_tok[:-4], dots=1)
    return duration.Duration(type=rhythm_tok)


def _rhythm_qlen(rhythm_tok: str) -> float:
    return float(_token_to_duration(rhythm_tok).quarterLength)


def _diatonic_pitch_tokens(fifths: int, lo: str, hi: str) -> tuple[str, ...]:
    ks = key.KeySignature(fifths)
    scale_obj = ks.getScale("major")
    pitches = scale_obj.getPitches(lo, hi)
    return tuple(_music21_pitch_to_token(p) for p in pitches)


class MelodyGenerator:
    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self._qlen_cache: dict[str, float] = {
            r: _rhythm_qlen(r) for r in self.config.rhythm_pool
        }
        self._weight_lookup: Mapping[str, float] = dict(
            zip(self.config.rhythm_pool, self.config.rhythm_weights)
        )

    def generate(self, seed: int, sample_idx: int) -> GeneratedSample:
        rng = np.random.default_rng(
            np.random.SeedSequence([int(seed), int(sample_idx)])
        )

        ts_str = str(rng.choice(self.config.time_signatures))
        clef_str = str(rng.choice(self.config.clefs))
        lo, hi = self.config.key_signature_range
        ks_fifths = int(rng.integers(lo, hi + 1))
        nb_lo, nb_hi = self.config.num_bars_range
        num_bars = int(rng.integers(nb_lo, nb_hi + 1))

        s = stream.Stream()
        s.append(clef.clefFromString(clef_str))
        s.append(key.KeySignature(ks_fifths))
        s.append(meter.TimeSignature(ts_str))

        labels: dict[str, list[str | None]] = {
            "type": [],
            "pitch": [],
            "rhythm": [],
            "attribute": [],
        }
        self._emit(labels, "clef", attribute=clef_str)
        self._emit(labels, "key_signature", attribute=f"ks{ks_fifths:+d}")
        self._emit(labels, "time_signature", attribute=ts_str)

        pitch_alphabet = _diatonic_pitch_tokens(
            ks_fifths, self.config.pitch_range[0], self.config.pitch_range[1]
        )
        if not pitch_alphabet:
            raise RuntimeError(
                f"Empty diatonic pitch alphabet for fifths={ks_fifths} "
                f"range={self.config.pitch_range}"
            )

        ts_obj = meter.TimeSignature(ts_str)
        bar_qlen = float(ts_obj.barDuration.quarterLength)

        for bar_idx in range(num_bars):
            self._fill_bar(s, labels, rng, bar_qlen, pitch_alphabet)
            if bar_idx < num_bars - 1:
                s.append(bar.Barline())
                self._emit(labels, "barline")

        return GeneratedSample(stream=s, labels=labels)

    def _fill_bar(
        self,
        s: stream.Stream,
        labels: dict[str, list[str | None]],
        rng: np.random.Generator,
        bar_qlen: float,
        pitch_alphabet: tuple[str, ...],
    ) -> None:
        remaining = bar_qlen
        cfg = self.config
        while remaining > _QLEN_EPS:
            valid = [
                r
                for r in cfg.rhythm_pool
                if self._qlen_cache[r] <= remaining + _QLEN_EPS
            ]
            if not valid:
                # Pool can't divide bar cleanly — emit a single residual rest.
                d = duration.Duration(quarterLength=remaining)
                s.append(note.Rest(duration=d))
                self._emit(labels, "rest", rhythm=cfg.rhythm_pool[-1])
                return

            weights = np.array(
                [self._weight_lookup[r] for r in valid], dtype=np.float64
            )
            weights /= weights.sum()
            r_tok = str(rng.choice(valid, p=weights))
            qlen = self._qlen_cache[r_tok]
            d = _token_to_duration(r_tok)

            if float(rng.random()) < cfg.rest_probability:
                s.append(note.Rest(duration=d))
                self._emit(labels, "rest", rhythm=r_tok)
            else:
                p_tok = str(rng.choice(pitch_alphabet))
                p = pitch.Pitch(_token_to_music21_pitch(p_tok))
                s.append(note.Note(p, duration=d))
                self._emit(labels, "note", pitch=p_tok, rhythm=r_tok)

            remaining -= qlen

    @staticmethod
    def _emit(
        labels: dict[str, list[str | None]],
        type_tok: str,
        pitch: str | None = None,
        rhythm: str | None = None,
        attribute: str | None = None,
    ) -> None:
        labels["type"].append(type_tok)
        labels["pitch"].append(pitch)
        labels["rhythm"].append(rhythm)
        labels["attribute"].append(attribute)
