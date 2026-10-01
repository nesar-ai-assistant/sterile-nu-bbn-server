"""Publication-style figure helpers shared by this server's plot tools.

Same conventions as the cosmic-emulator server's figures:
- NO system LaTeX (hosted VMs have none): matplotlib mathtext with the
  STIX fonts that ship with matplotlib gives the journal look.
- Nothing may overflow the canvas: short axis labels, wrapped titles,
  legends outside the axes when they are long, constrained layout plus a
  tight bounding box at save time.
- A fixed-order, colorblind-validated categorical palette, always paired
  with distinct linestyles so identity is never color-alone.
"""

import re
from pathlib import Path

# Okabe-Ito-derived order; adjacent-pair CVD dE >= 11 (validated).
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#882255",
           "#B8860B", "#CC79A7", "#6B4C9A", "#E07B39"]
LINESTYLES = ["-", "--", "-.", ":", (0, (6, 1.5, 1, 1.5, 1, 1.5)),
              (0, (1, 1)), (0, (8, 2)), (0, (3, 1, 1, 1))]
INK = "#222222"
MUTED = "#6b6b6b"
BAND_ALPHA = 0.22


def rc_params(base_size: float = 12.0) -> dict:
    return {
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": base_size,
        "axes.labelsize": base_size + 1,
        "axes.titlesize": base_size + 1,
        "legend.fontsize": base_size - 1.5,
        "xtick.labelsize": base_size - 0.5,
        "ytick.labelsize": base_size - 0.5,
        "axes.linewidth": 0.9,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 2.5,
        "ytick.minor.size": 2.5,
        "lines.linewidth": 1.8,
        "legend.frameon": False,
        "legend.handlelength": 2.6,
        "savefig.dpi": 200,
        "figure.dpi": 100,
        "axes.unicode_minus": True,
    }


def wrap(text: str, width: int) -> str:
    """Wrap text without breaking inside $...$ math spans."""
    if len(text) <= width:
        return text
    tokens = re.findall(r"\$[^$]*\$|\S+", text)
    lines, current = [], ""
    for tok in tokens:
        candidate = f"{current} {tok}".strip()
        visible = len(re.sub(r"\\[a-zA-Z]+|[{}$^_\\]", "", candidate))
        if visible > width and current:
            lines.append(current)
            current = tok
        else:
            current = candidate
    lines.append(current)
    return "\n".join(lines)


def legend_below(fig, handles, labels, ncol: int = 2, title: str | None = None):
    """Legend outside the axes, under the figure (never covers data)."""
    leg = fig.legend(handles, [wrap(lab, 46) for lab in labels],
                     loc="outside lower center", ncol=ncol,
                     borderaxespad=0.2, title=title)
    if title:
        leg.get_title().set_fontsize(10)
        leg.get_title().set_color(MUTED)
    return leg


def save(fig, path: Path, save_pdf: bool = False) -> list[str]:
    """Save PNG (and optionally a vector PDF next to it); return paths."""
    out = [str(path)]
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    if save_pdf:
        pdf = path.with_suffix(".pdf")
        fig.savefig(pdf, bbox_inches="tight", pad_inches=0.08)
        out.append(str(pdf))
    return out
