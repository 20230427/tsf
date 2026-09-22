#!/usr/bin/env python3
"""Draw the DD-Mamba structure diagram in the style of the supplied Visio
template and save it as editable .vsdx.

Output goes to ``diagrams/`` and is deliberately SEPARATE from the manuscript
figures. The paper keeps its own schematics (fig_architecture.pdf,
fig_selective_blocks.pdf, produced by draw_scientific_figures.py); nothing here
writes into paper/neurocomputing/, so re-running this can never overwrite them.

  diagrams/dd_mamba_structure.vsdx   3 pages, editable in Visio
  diagrams/preview_*.pdf/.png        rendering of the same pages, for review

Pages:
  1  end-to-end DD-Mamba pipeline
  2  forecast fusion and reconstruction
  3  the three ablatable blocks (TD / VC / FD)

A .vsdx is an OPC package -- a ZIP of XML parts, same container family as
.docx -- so it is written directly, with no Visio and no Windows involved. The
preview renderer consumes the same Diagram objects, so the previews cannot
drift from the Visio source.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from vsdx_lib import (AMBER, BLUE, CYAN, GREEN, GREY, INK, PINK, WHITE, YELLOW,
                      LW_THIN, Diagram, render_pdf, write_vsdx)

OUT = ROOT / "diagrams"


def architecture() -> Diagram:
    """End-to-end pipeline, top-down: RevIN -> two branches -> per-branch
    forecasts. Fusion continues in the second panel."""
    d = Diagram(width=7.2, height=5.5, title="DD-Mamba architecture")
    TX, FX = 1.95, 5.25          # branch centre lines
    MID = 3.60

    # Domain bands (drawn first so blocks sit on top).
    d.box(TX, 2.52, 3.10, 3.80, "", fill=GREY, dashed=True, lw=LW_THIN,
          rounding=0.08, z=1)
    d.box(FX, 2.52, 3.10, 3.80, "", fill=GREY, dashed=True, lw=LW_THIN,
          rounding=0.08, z=1)

    # Input and normalization.
    d.box(MID, 5.26, 2.6, 0.34, "Input window  X  (B, L, C)", fill=BLUE)
    d.box(MID, 4.72, 2.9, 0.40,
          "RevIN / alpha-RevIN\\nx = a(X - mu)/sigma + (1 - a)X", fill=PINK,
          fontsize=8)

    # Band titles, inside the band at its top edge.
    d.box(TX, 4.28, 3.00, 0.26, "Time domain", fill=CYAN, fontsize=8.5,
          bold=True, lw=LW_THIN)
    d.box(FX, 4.28, 3.00, 0.26, "Frequency domain", fill=YELLOW, fontsize=8.5,
          bold=True, lw=LW_THIN)

    # ---- time branch, descending ----
    d.box(TX, 3.80, 2.7, 0.36, "Series decomposition (k = 25)", fill=CYAN,
          fontsize=8.5)
    d.box(TX - 0.68, 3.22, 1.20, 0.30, "Trend", fill=WHITE, fontsize=8.5)
    d.box(TX + 0.68, 3.22, 1.20, 0.30, "Seasonal", fill=WHITE, fontsize=8.5)
    d.box(TX, 2.66, 2.7, 0.32, "Linear backbone (DLinear)", fill=CYAN,
          fontsize=8.5)
    d.box(TX, 2.06, 2.7, 0.38, "Temporal encoder  [TD]\\nMamba  |  MLP",
          fill=CYAN, fontsize=8)
    d.box(TX, 1.42, 2.7, 0.38, "Variate mixer  [VC]\\nbi-Mamba over channels",
          fill=AMBER, fontsize=8)
    d.box(TX, 0.82, 2.7, 0.32, "forecast head", fill=WHITE, fontsize=8.5)
    d.box(TX, 0.26, 1.6, 0.32, "y_time", fill=CYAN, fontsize=9.5, bold=True)

    # ---- frequency branch, descending ----
    d.box(FX, 3.80, 2.7, 0.32, "rFFT  ->  spectrum Z", fill=YELLOW, fontsize=8.5)
    d.box(FX, 3.26, 2.7, 0.30, "low-pass (rho)", fill=WHITE, fontsize=8.5)
    d.box(FX, 2.66, 2.7, 0.38,
          "Spectral encoder  [FD]\\ncomplex filter  |  bi-Mamba", fill=YELLOW,
          fontsize=8)
    d.box(FX, 2.06, 2.7, 0.38, "FITS map (optional)\\nzero-init", fill=WHITE,
          fontsize=8, dashed=True)
    d.box(FX, 1.42, 2.7, 0.38, "Variate mixer  [VC]", fill=AMBER, fontsize=8)
    d.box(FX, 0.82, 2.7, 0.32, "forecast head", fill=WHITE, fontsize=8.5)
    d.box(FX, 0.26, 1.6, 0.32, "y_freq", fill=YELLOW, fontsize=9.5, bold=True)

    # ---- wiring ----
    d.arrow(MID, 5.09, MID, 4.92)
    d.elbow(MID, 4.52, TX, 3.98, first="v")
    d.elbow(MID, 4.52, FX, 3.98, first="v")
    d.elbow(TX, 3.62, TX - 0.68, 3.37, first="v")
    d.elbow(TX, 3.62, TX + 0.68, 3.37, first="v")
    d.elbow(TX - 0.68, 3.07, TX, 2.82, first="v")
    d.elbow(TX + 0.68, 3.07, TX, 2.82, first="v")
    for y0, y1 in [(2.50, 2.25), (1.87, 1.61), (1.23, 0.98), (0.66, 0.42)]:
        d.arrow(TX, y0, TX, y1)
    d.arrow(FX, 3.64, FX, 3.41)
    d.arrow(FX, 3.11, FX, 2.85)
    d.arrow(FX, 2.47, FX, 2.25, dashed=True)
    d.arrow(FX, 1.87, FX, 1.61)
    d.arrow(FX, 1.23, FX, 0.98)
    d.arrow(FX, 0.66, FX, 0.42)
    return d


def architecture_lower() -> Diagram:
    """Fusion and reconstruction (second panel)."""
    d = Diagram(width=7.2, height=1.75, title="DD-Mamba fusion")
    d.box(1.55, 1.48, 1.7, 0.32, "y_time", fill=CYAN, fontsize=9.5, bold=True)
    d.box(5.65, 1.48, 1.7, 0.32, "y_freq", fill=YELLOW, fontsize=9.5, bold=True)
    d.box(3.60, 1.48, 2.1, 0.34, "gate  g = sigmoid(MLP[f_t, f_f])", fill=GREEN,
          fontsize=8)
    d.box(3.60, 0.88, 4.2, 0.38,
          "Forecast fusion\\nconvex:  g y_t + (1-g) y_f      additive:  y_t + a y_f",
          fill=GREEN, fontsize=8)
    d.box(3.60, 0.26, 2.8, 0.32, "de-normalize  ->  Y  (B, H, C)", fill=BLUE,
          fontsize=9)
    d.elbow(1.55, 1.32, 2.90, 1.07, first="v")
    d.elbow(5.65, 1.32, 4.30, 1.07, first="v")
    d.arrow(3.60, 1.31, 3.60, 1.07)
    d.arrow(3.60, 0.69, 3.60, 0.42)
    return d


def selective_blocks() -> Diagram:
    """The three ablatable blocks, matching the ablation grid in the paper."""
    d = Diagram(width=7.2, height=3.3, title="Ablatable blocks")

    # (a) temporal dependency block
    d.box(1.20, 3.05, 2.15, 0.26, "(a)  TD  temporal encoder", fill=CYAN,
          fontsize=8.5, bold=True)
    d.box(1.20, 2.45, 2.15, 0.32, "RMSNorm", fill=WHITE, fontsize=8.5)
    d.box(1.20, 1.85, 2.15, 0.34, "Mamba  (selective SSM)\\ncausal over time",
          fill=CYAN, fontsize=8)
    d.box(1.20, 1.22, 2.15, 0.32, "FFN", fill=WHITE, fontsize=8.5)
    d.box(1.20, 0.55, 2.15, 0.32, "swap: MLP  |  w/o", fill=GREY, fontsize=8,
          dashed=True)
    d.arrow(1.20, 2.29, 1.20, 2.02)
    d.arrow(1.20, 1.68, 1.20, 1.38)
    d.arrow(1.20, 1.06, 1.20, 0.71, dashed=True)

    # (b) variate-correlation block
    d.box(3.60, 3.05, 2.15, 0.26, "(b)  VC  variate mixer", fill=AMBER,
          fontsize=8.5, bold=True)
    d.box(3.60, 2.45, 2.15, 0.32, "forward scan", fill=WHITE, fontsize=8.5)
    d.box(3.60, 1.85, 2.15, 0.32, "backward scan", fill=WHITE, fontsize=8.5)
    d.box(3.60, 1.22, 2.15, 0.34, "sum  +  FFN\\n(bidirectional)", fill=AMBER,
          fontsize=8)
    d.box(3.60, 0.55, 2.15, 0.32, "swap: uni-Mamba | Attention | w/o",
          fill=GREY, fontsize=7.5, dashed=True)
    d.arrow(3.60, 2.29, 3.60, 2.02)
    d.arrow(3.60, 1.68, 3.60, 1.40)
    d.arrow(3.60, 1.05, 3.60, 0.71, dashed=True)

    # (c) frequency-domain block
    d.box(6.00, 3.05, 2.15, 0.26, "(c)  FD  frequency branch", fill=YELLOW,
          fontsize=8.5, bold=True)
    d.box(6.00, 2.45, 2.15, 0.32, "rFFT  ->  (re, im)", fill=WHITE, fontsize=8.5)
    d.box(6.00, 1.85, 2.15, 0.34, "complex linear filter\\nover bins", fill=YELLOW,
          fontsize=8)
    d.box(6.00, 1.22, 2.15, 0.32, "forecast head", fill=WHITE, fontsize=8.5)
    d.box(6.00, 0.55, 2.15, 0.32, "swap: bi-Mamba | w/o", fill=GREY,
          fontsize=8, dashed=True)
    d.arrow(6.00, 2.29, 6.00, 2.02)
    d.arrow(6.00, 1.68, 6.00, 1.38)
    d.arrow(6.00, 1.06, 6.00, 0.71, dashed=True)

    return d


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    arch = architecture()
    fuse = architecture_lower()
    blocks = selective_blocks()

    vsdx = OUT / "dd_mamba_structure.vsdx"
    write_vsdx([arch, fuse, blocks], str(vsdx))
    render_pdf(arch, str(OUT / "preview_1_architecture.pdf"))
    render_pdf(fuse, str(OUT / "preview_2_fusion.pdf"))
    render_pdf(blocks, str(OUT / "preview_3_blocks.pdf"))
    print(f"[saved] {vsdx.relative_to(ROOT)}  (3 pages)")
    print("[saved] diagrams/preview_{1_architecture,2_fusion,3_blocks}.pdf")
    print("note: the manuscript's own figures are untouched.")


if __name__ == "__main__":
    main()
