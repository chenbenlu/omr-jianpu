from __future__ import annotations

import io
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from music21 import chord, stream
from music21.musicxml.m21ToXml import GeneralObjectExporter
from PIL import Image


@dataclass(frozen=True)
class RenderConfig:
    page_width: int = 2100
    page_height: int = 400
    page_margin_top: int = 50
    page_margin_left: int = 50
    page_margin_right: int = 50
    page_margin_bottom: int = 50
    scale: int = 40
    adjust_page_height: bool = True
    breaks: str = "none"
    output_width: int | None = None
    output_height: int | None = None
    extra_options: dict[str, Any] = field(default_factory=dict)


def _build_verovio_options(cfg: RenderConfig) -> dict[str, Any]:
    opts: dict[str, Any] = {
        "pageWidth": cfg.page_width,
        "pageHeight": cfg.page_height,
        "pageMarginTop": cfg.page_margin_top,
        "pageMarginLeft": cfg.page_margin_left,
        "pageMarginRight": cfg.page_margin_right,
        "pageMarginBottom": cfg.page_margin_bottom,
        "scale": cfg.scale,
        "adjustPageHeight": cfg.adjust_page_height,
        "breaks": cfg.breaks,
        "footer": "none",
        "header": "none",
    }
    opts.update(cfg.extra_options)
    return opts


class StaffRenderer:
    def __init__(self, config: RenderConfig | None = None) -> None:
        self.config = config or RenderConfig()
        # verovio toolkits and cairosvg state aren't picklable across fork/spawn;
        # the toolkit is built lazily on first use so DataLoader workers get a
        # fresh one per process.
        self._toolkit: Any | None = None

    def _ensure_toolkit(self) -> Any:
        if self._toolkit is None:
            import verovio  # lazy: keeps `import src.data` cheap for mocked tests

            tk = verovio.toolkit()
            tk.setOptions(_build_verovio_options(self.config))
            self._toolkit = tk
        return self._toolkit

    def render(self, score: stream.Stream) -> np.ndarray:
        if any(isinstance(e, chord.Chord) for e in score.recurse()):
            raise ValueError(
                "Renderer received a Stream containing a Chord; monophonic only"
            )

        xml_str = GeneralObjectExporter(score).parse().decode("utf-8")
        tk = self._ensure_toolkit()
        if not tk.loadData(xml_str):
            raise RuntimeError("verovio failed to parse the MusicXML payload")
        svg = tk.renderToSVG(1)
        return _svg_to_rgb(svg, self.config.output_width, self.config.output_height)

    def __getstate__(self) -> dict[str, Any]:
        # Strip the live toolkit before pickling (DataLoader worker fork path).
        state = self.__dict__.copy()
        state["_toolkit"] = None
        return state


def _svg_to_rgb(
    svg: str, output_width: int | None, output_height: int | None
) -> np.ndarray:
    import cairosvg  # lazy: tests mock StaffRenderer.render and never hit this

    png_bytes = cairosvg.svg2png(
        bytestring=svg.encode("utf-8"),
        output_width=output_width,
        output_height=output_height,
    )
    # verovio's SVG has a transparent background; PIL's RGBA->RGB convert composites
    # onto black, which would give us an all-black image. Composite onto white so
    # the unpainted page maps to 255 (and to +1.0 after Normalize(0.5, 0.5)).
    rgba = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    img = Image.alpha_composite(white, rgba).convert("RGB")
    return np.asarray(img, dtype=np.uint8)
