#!/usr/bin/env python3
"""Minimal VSDX (Visio 2013+) writer, plus a matplotlib renderer of the same
layout so the editable source and the figure in the paper cannot drift apart.

Why write VSDX by hand: a .vsdx is an OPC package (a ZIP of XML parts), the
same container family as .docx. Producing one needs no Visio and no Windows,
only correct parts and relationships.

Why also render with matplotlib: LaTeX cannot \\includegraphics a .vsdx, and
LibreOffice fails to import these files. Rather than maintain two drawings,
both back-ends consume one ``Diagram`` object, so the PDF in the manuscript is
by construction the same picture as the editable Visio source.

Style constants below are measured from the drawing template supplied with the
request: pastel fills, thin black outlines, Times New Roman, 1 mm corner
rounding.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# ---------------------------------------------------------------- style ----
# Fill palette, taken from the template's most-used FillForegnd values.
GREY = "#d8d8d8"    # containers / background bands
CYAN = "#cbffff"    # time-domain blocks
YELLOW = "#fff2cc"  # frequency-domain blocks
GREEN = "#c8fdd3"   # fusion / output
PINK = "#ffcccc"    # normalization
BLUE = "#deebf6"    # inputs
WHITE = "#ffffff"
AMBER = "#fee599"   # emphasis

INK = "#000000"
FONT = "Times New Roman"
LW_THIN = 0.003472222222222222   # template default (~0.25 pt)
LW_NORM = 0.01388888888888889    # ~1 pt
ROUND = 0.03937007874015748      # 1 mm


@dataclass
class Box:
    x: float          # centre, inches
    y: float
    w: float
    h: float
    text: str = ""
    fill: str = CYAN
    rounding: float = ROUND
    fontsize: float = 9.0
    bold: bool = False
    dashed: bool = False
    line: str = INK
    lw: float = LW_NORM
    z: int = 3            # bands use z=1 so connectors (z=2) stay visible


@dataclass
class Arrow:
    x1: float
    y1: float
    x2: float
    y2: float
    text: str = ""
    dashed: bool = False
    line: str = INK
    lw: float = LW_NORM
    head: bool = True      # mid-segments of an elbow carry no arrowhead


@dataclass
class Diagram:
    width: float                       # page size, inches
    height: float
    boxes: List[Box] = field(default_factory=list)
    arrows: List[Arrow] = field(default_factory=list)
    title: str = "Diagram"

    def box(self, *a, **kw) -> Box:
        b = Box(*a, **kw)
        self.boxes.append(b)
        return b

    def arrow(self, *a, **kw) -> Arrow:
        ar = Arrow(*a, **kw)
        self.arrows.append(ar)
        return ar

    # ---- orthogonal routing helper: down-then-across, or across-then-down --
    def elbow(self, x1, y1, x2, y2, first: str = "v", **kw):
        """Two-segment orthogonal connection; only the last segment is arrowed.

        Journals expect right-angled routing rather than diagonals, which is
        also what the supplied template uses throughout.
        """
        if first == "v":
            mid = (x1, y2)
        else:
            mid = (x2, y1)
        self.arrows.append(Arrow(x1, y1, mid[0], mid[1], head=False,
                                 line=kw.get("line", INK),
                                 dashed=kw.get("dashed", False),
                                 lw=kw.get("lw", LW_NORM)))
        self.arrows.append(Arrow(mid[0], mid[1], x2, y2, **kw))


# ------------------------------------------------------------------ vsdx ----
_NS = "http://schemas.microsoft.com/office/visio/2012/main"
_RNS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _esc(t: str) -> str:
    return (t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
             .replace('"', "&quot;"))


def _shape_rect(sid: int, b: Box) -> str:
    """A rounded rectangle with its own geometry (no master needed)."""
    dash = "<Cell N='LinePattern' V='2'/>" if b.dashed else ""
    txt = ""
    if b.text:
        # Visio stores newlines literally inside <Text>.
        txt = "<Text>%s</Text>" % _esc(b.text).replace("\\n", "\n")
    return f"""<Shape ID='{sid}' NameU='Box{sid}' Type='Shape' LineStyle='0' FillStyle='0' TextStyle='0'>
<Cell N='PinX' V='{b.x}'/><Cell N='PinY' V='{b.y}'/>
<Cell N='Width' V='{b.w}'/><Cell N='Height' V='{b.h}'/>
<Cell N='LocPinX' V='{b.w/2}' F='Width*0.5'/><Cell N='LocPinY' V='{b.h/2}' F='Height*0.5'/>
<Cell N='Angle' V='0'/><Cell N='FlipX' V='0'/><Cell N='FlipY' V='0'/>
<Cell N='FillForegnd' V='{b.fill}'/><Cell N='FillPattern' V='1'/>
<Cell N='LineColor' V='{b.line}'/><Cell N='LineWeight' V='{b.lw}'/>
<Cell N='Rounding' V='{b.rounding}'/>{dash}
<Cell N='VerticalAlign' V='1'/><Cell N='TextBkgnd' V='0'/>
<Section N='Character'><Row IX='0'>
<Cell N='Font' V='{FONT}'/><Cell N='Color' V='{INK}'/>
<Cell N='Size' V='{b.fontsize/72.0}'/><Cell N='Style' V='{1 if b.bold else 0}'/>
</Row></Section>
<Section N='Paragraph'><Row IX='0'><Cell N='HorzAlign' V='1'/></Row></Section>
<Section N='Geometry' IX='0'>
<Cell N='NoFill' V='0'/><Cell N='NoLine' V='0'/><Cell N='NoShow' V='0'/>
<Row T='RelMoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>
<Row T='RelLineTo' IX='2'><Cell N='X' V='1'/><Cell N='Y' V='0'/></Row>
<Row T='RelLineTo' IX='3'><Cell N='X' V='1'/><Cell N='Y' V='1'/></Row>
<Row T='RelLineTo' IX='4'><Cell N='X' V='0'/><Cell N='Y' V='1'/></Row>
<Row T='RelLineTo' IX='5'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>
</Section>{txt}</Shape>"""


def _shape_arrow(sid: int, a: Arrow) -> str:
    x0, y0 = min(a.x1, a.x2), min(a.y1, a.y2)
    w, h = a.x2 - a.x1, a.y2 - a.y1
    dash = "<Cell N='LinePattern' V='2'/>" if a.dashed else ""
    head = ("<Cell N='EndArrow' V='4'/><Cell N='EndArrowSize' V='1'/>"
            if a.head else "")
    txt = "<Text>%s</Text>" % _esc(a.text) if a.text else ""
    # Geometry is expressed relative to the shape's own box; use absolute
    # MoveTo/LineTo in local coordinates with the pin at the start point.
    return f"""<Shape ID='{sid}' NameU='Conn{sid}' Type='Shape' LineStyle='0' FillStyle='0' TextStyle='0'>
<Cell N='PinX' V='{a.x1}'/><Cell N='PinY' V='{a.y1}'/>
<Cell N='Width' V='{abs(w) if w else 0.0001}'/><Cell N='Height' V='{abs(h) if h else 0.0001}'/>
<Cell N='LocPinX' V='0'/><Cell N='LocPinY' V='0'/>
<Cell N='LineColor' V='{a.line}'/><Cell N='LineWeight' V='{a.lw}'/>
{head}{dash}
<Cell N='FillPattern' V='0'/>
<Section N='Character'><Row IX='0'><Cell N='Font' V='{FONT}'/>
<Cell N='Size' V='{7.5/72.0}'/><Cell N='Color' V='{INK}'/></Row></Section>
<Section N='Geometry' IX='0'>
<Cell N='NoFill' V='1'/><Cell N='NoLine' V='0'/><Cell N='NoShow' V='0'/>
<Row T='MoveTo' IX='1'><Cell N='X' V='0'/><Cell N='Y' V='0'/></Row>
<Row T='LineTo' IX='2'><Cell N='X' V='{w}'/><Cell N='Y' V='{h}'/></Row>
</Section>{txt}</Shape>"""


def write_vsdx(diagrams: List[Diagram], path: str) -> None:
    """Write one page per diagram into a single .vsdx package."""
    n = len(diagrams)
    ct = ["<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
          "<Types xmlns='http://schemas.openxmlformats.org/package/2006/content-types'>",
          "<Default Extension='rels' ContentType='application/vnd.openxmlformats-package.relationships+xml'/>",
          "<Default Extension='xml' ContentType='application/xml'/>",
          "<Override PartName='/visio/document.xml' ContentType='application/vnd.ms-visio.drawing.main+xml'/>",
          "<Override PartName='/visio/pages/pages.xml' ContentType='application/vnd.ms-visio.pages+xml'/>"]
    for i in range(1, n + 1):
        ct.append(f"<Override PartName='/visio/pages/page{i}.xml' "
                  f"ContentType='application/vnd.ms-visio.page+xml'/>")
    ct += ["<Override PartName='/visio/windows.xml' ContentType='application/vnd.ms-visio.windows+xml'/>",
           "<Override PartName='/docProps/core.xml' ContentType='application/vnd.openxmlformats-package.core-properties+xml'/>",
           "<Override PartName='/docProps/app.xml' ContentType='application/vnd.openxmlformats-officedocument.extended-properties+xml'/>",
           "</Types>"]

    root_rels = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
<Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/document' Target='visio/document.xml'/>
<Relationship Id='rId2' Type='{_RNS}/metadata/core-properties' Target='docProps/core.xml'/>
<Relationship Id='rId3' Type='{_RNS}/extended-properties' Target='docProps/app.xml'/>
</Relationships>"""

    doc = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<VisioDocument xmlns='{_NS}' xmlns:r='{_RNS}' xml:space='preserve'>
<DocumentSettings TopPage='0' DefaultTextStyle='0' DefaultLineStyle='0' DefaultFillStyle='0'>
<GlyphSettings V='0'/></DocumentSettings>
<Colors/><FaceNames><FaceName NameU='{FONT}'/></FaceNames>
<StyleSheets/></VisioDocument>"""

    doc_rels = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>
<Relationship Id='rId1' Type='http://schemas.microsoft.com/visio/2010/relationships/pages' Target='pages/pages.xml'/>
<Relationship Id='rId2' Type='http://schemas.microsoft.com/visio/2010/relationships/windows' Target='windows.xml'/>
</Relationships>"""

    pages = ["<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
             f"<Pages xmlns='{_NS}' xmlns:r='{_RNS}' xml:space='preserve'>"]
    for i, d in enumerate(diagrams):
        pages.append(
            f"<Page ID='{i}' NameU='{_esc(d.title)}' Name='{_esc(d.title)}' ViewScale='1' "
            f"ViewCenterX='{d.width/2}' ViewCenterY='{d.height/2}'>"
            f"<PageSheet LineStyle='0' FillStyle='0' TextStyle='0'>"
            f"<Cell N='PageWidth' V='{d.width}'/><Cell N='PageHeight' V='{d.height}'/>"
            f"<Cell N='PageScale' V='1'/><Cell N='DrawingScale' V='1'/>"
            f"<Cell N='DrawingSizeType' V='3'/><Cell N='DrawingScaleType' V='0'/>"
            f"</PageSheet><Rel r:id='rId{i+1}'/></Page>")
    pages.append("</Pages>")

    pages_rels = ["<?xml version='1.0' encoding='UTF-8' standalone='yes'?>",
                  "<Relationships xmlns='http://schemas.openxmlformats.org/package/2006/relationships'>"]
    for i in range(1, n + 1):
        pages_rels.append(
            f"<Relationship Id='rId{i}' "
            f"Type='http://schemas.microsoft.com/visio/2010/relationships/page' "
            f"Target='page{i}.xml'/>")
    pages_rels.append("</Relationships>")

    windows = f"""<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Windows xmlns='{_NS}' xmlns:r='{_RNS}' ClientWidth='1200' ClientHeight='800'/>"""

    core = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<cp:coreProperties xmlns:cp='http://schemas.openxmlformats.org/package/2006/metadata/core-properties'
 xmlns:dc='http://purl.org/dc/elements/1.1/'>
<dc:title>DD-Mamba figures</dc:title><dc:creator>DD-Mamba</dc:creator>
</cp:coreProperties>"""

    app = """<?xml version='1.0' encoding='UTF-8' standalone='yes'?>
<Properties xmlns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'>
<Application>Microsoft Visio</Application></Properties>"""

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", "".join(ct))
        z.writestr("_rels/.rels", root_rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("docProps/app.xml", app)
        z.writestr("visio/document.xml", doc)
        z.writestr("visio/_rels/document.xml.rels", doc_rels)
        z.writestr("visio/windows.xml", windows)
        z.writestr("visio/pages/pages.xml", "".join(pages))
        z.writestr("visio/pages/_rels/pages.xml.rels", "".join(pages_rels))
        for i, d in enumerate(diagrams, start=1):
            shapes, sid = [], 1
            for b in d.boxes:
                shapes.append(_shape_rect(sid, b)); sid += 1
            for a in d.arrows:
                shapes.append(_shape_arrow(sid, a)); sid += 1
            page = (f"<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
                    f"<PageContents xmlns='{_NS}' xmlns:r='{_RNS}' xml:space='preserve'>"
                    f"<Shapes>{''.join(shapes)}</Shapes></PageContents>")
            z.writestr(f"visio/pages/page{i}.xml", page)


# -------------------------------------------------------------- rendering ----
def render_pdf(d: Diagram, path: str, dpi: int = 300) -> None:
    """Render the same Diagram with matplotlib, for \\includegraphics."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(d.width, d.height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, d.width); ax.set_ylim(0, d.height)
    ax.axis("off")

    for a in d.arrows:
        ax.add_patch(FancyArrowPatch(
            (a.x1, a.y1), (a.x2, a.y2),
            arrowstyle="-|>" if a.head else "-", mutation_scale=9,
            linewidth=a.lw * 72, color=a.line,
            linestyle=(0, (4, 2)) if a.dashed else "solid",
            shrinkA=0, shrinkB=0, zorder=2))
        if a.text:
            ax.text((a.x1 + a.x2) / 2, (a.y1 + a.y2) / 2, a.text,
                    fontsize=7.5, ha="center", va="bottom", color=INK, zorder=4)

    for b in sorted(d.boxes, key=lambda b: b.z):
        r = min(b.rounding, b.w / 2, b.h / 2)
        ax.add_patch(FancyBboxPatch(
            (b.x - b.w / 2 + r, b.y - b.h / 2 + r),
            b.w - 2 * r, b.h - 2 * r,
            boxstyle=f"round,pad={r}",
            facecolor=b.fill, edgecolor=b.line,
            linewidth=b.lw * 72,
            linestyle=(0, (4, 2)) if b.dashed else "solid", zorder=b.z))
        if b.text:
            ax.text(b.x, b.y, b.text.replace("\\n", "\n"),
                    ha="center", va="center", fontsize=b.fontsize,
                    fontweight="bold" if b.bold else "normal",
                    color=INK, zorder=b.z + 1, linespacing=1.25)

    fig.savefig(path, format="pdf", bbox_inches="tight", pad_inches=0.02)
    png = path[:-4] + ".png" if path.endswith(".pdf") else path + ".png"
    fig.savefig(png, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
