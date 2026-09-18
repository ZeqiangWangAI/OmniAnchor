"""Rebuild the supplied figure as editable SVG; no model scores are fabricated.

Only Python's standard library is required. All artwork is made from SVG text,
paths, and shapes; the two photo placements embed the documented OASIS asset.
"""
from __future__ import annotations

import argparse
import base64
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

HERE = Path(__file__).resolve().parent
SVG = "http://www.w3.org/2000/svg"
# Inkscape uses this historical namespace URI.
INK = "http://www.inkscape.org/namespaces/inkscape"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("inkscape", INK)
ET.register_namespace("xlink", XLINK)
BLUE = "#005bdc"
DARKBLUE = "#002d65"
INKCOLOR = "#080d16"
MUTED = "#48515e"
BORDER = "#dbe2e9"
PHOTO = "assets/oasis_I466_Lake_12_native.jpg"
WIDTH, HEIGHT = 1536, 1072


def el(parent, tag, **attrs):
    attrs = {k.replace("_", "-"): str(v) for k, v in attrs.items() if v is not None}
    return ET.SubElement(parent, f"{{{SVG}}}{tag}", attrs)


def group(parent, name, layer=False):
    g = el(parent, "g", id=name)
    g.set(f"{{{INK}}}label", name.replace("-", " "))
    if layer:
        g.set(f"{{{INK}}}groupmode", "layer")
    return g


def rect(p, x, y, w, h, fill="white", stroke=BORDER, sw=1, r=4, **kw):
    return el(p, "rect", x=x, y=y, width=w, height=h, rx=r, fill=fill,
              stroke=stroke, stroke_width=sw, **kw)


def line(p, x1, y1, x2, y2, color=INKCOLOR, sw=1.5, **kw):
    return el(p, "line", x1=x1, y1=y1, x2=x2, y2=y2, stroke=color,
              stroke_width=sw, **kw)


def path(p, d, color=INKCOLOR, sw=1.5, fill="none", **kw):
    return el(p, "path", d=d, fill=fill, stroke=color, stroke_width=sw, **kw)


def text(p, x, y, value, size=16, bold=False, italic=False, color=INKCOLOR,
         anchor="start", rich=False, **kw):
    t = el(p, "text", x=x, y=y, font_size=size, fill=color, text_anchor=anchor,
           font_weight="bold" if bold else None, font_style="italic" if italic else None,
           **kw)
    if rich:
        fragment = ET.fromstring(f'<text xmlns="{SVG}">{value}</text>')
        t.text = fragment.text
        for child in fragment:
            t.append(child)
    else:
        t.text = value
    return t


def it(s):
    return '<tspan font-style="italic">' + escape(s) + '</tspan>'


def sub(s, amount=4):
    return f'<tspan dy="{amount}" font-size="70%">{escape(s)}</tspan><tspan dy="{-amount}">\u200b</tspan>'


def m(base, index=None):
    return it(base) + (sub(index) if index else "")


def score(i="i", b=True, x="x", bar=False):
    return m("s̄" if bar else "s", f"{i},b" if b else i) + "(" + it(x) + ")"


def arrow(p, x1, y1, x2, y2, sw=1.6, double=False, color=INKCOLOR):
    line(p, x1, y1, x2, y2, color, sw, marker_end="url(#arrow)",
         marker_start="url(#arrow-start)" if double else None)


def icon(p, name, x, y, width, color=INKCOLOR, stroke=1.6):
    g = group(p, "icon-" + name + "-" + str(x))
    g.set("transform", f"translate({x} {y}) scale({width / 24})")
    g.set("fill", "none")
    g.set("stroke", color)
    g.set("stroke-width", str(stroke))
    g.set("stroke-linecap", "round")
    g.set("stroke-linejoin", "round")
    source = ET.parse(HERE / "assets" / "icons" / f"{name}.svg").getroot()
    for child in source:
        g.append(copy.deepcopy(child))
    return g


def photo(p, x, y, w, h, name, fit="xMidYMid slice"):
    g = group(p, name)
    t = el(g, "title")
    t.text = "OASIS stimulus I466, Lake 12; photograph by James Wheeler. See SOURCES.md."
    im = el(g, "image", x=x, y=y, width=w, height=h, preserveAspectRatio=fit,
            id=name + "-replaceable-photo")
    im.set(f"{{{XLINK}}}href", "@asset:" + PHOTO)
    return g


def brace(p, x, y, h):
    # A single editable cubic path, opening toward the labels on its left.
    half = h / 2
    path(p, f"M {x} {y} C {x+8} {y} {x+8} {y+2} {x+8} {y+10} "
         f"L {x+8} {y+half-10} Q {x+8} {y+half} {x+17} {y+half} "
         f"Q {x+8} {y+half} {x+8} {y+half+10} L {x+8} {y+h-10} "
         f"C {x+8} {y+h-2} {x+8} {y+h} {x} {y+h}", sw=1.3)


def table(p, x, y, widths, headers, rows, row_h=28.4, fs=16, header_fs=15):
    w = sum(widths)
    h = row_h * (len(rows) + 1)
    rect(p, x, y, w, h, fill="white", stroke="#cbd1d8", sw=.85, r=0)
    rect(p, x, y, w, row_h, fill="url(#table-head)", stroke="#b7c1cb", sw=.85, r=0)
    for i in range(1, len(rows) + 1):
        line(p, x, y + i * row_h, x + w, y + i * row_h, "#cbd1d8", .8)
    dx = x
    for j, width in enumerate(widths):
        if j:
            line(p, dx, y, dx, y + h, "#bfc7cf", .8)
        text(p, dx + width / 2, y + row_h / 2 + header_fs * .34,
             headers[j], header_fs, bold=True, anchor="middle", rich=True)
        for i, row in enumerate(rows):
            aligned_left = j == 0 and fs >= 15
            text(p, dx + 15 if aligned_left else dx + width / 2,
                 y + (i + 1.5) * row_h + fs * .34, row[j], fs,
                 anchor="start" if aligned_left else "middle", rich=True)
        dx += width


def build():
    root = ET.Element(f"{{{SVG}}}svg", {
        "width": str(WIDTH), "height": str(HEIGHT), "viewBox": f"0 0 {WIDTH} {HEIGHT}",
        "version": "1.1", "font-family": "'Times New Roman', Times, 'Liberation Serif', serif",
        "shape-rendering": "geometricPrecision", "text-rendering": "optimizeLegibility",
    })
    el(root, "title").text = "OmniAnchor: scored event, named coordinates, and two validation routes"
    el(root, "desc").text = ("Editable reconstruction of the user-provided figure. "
        "All text, formula components, tables, connectors, icons and scatter points are vector objects. "
        "Two editable image objects show the documented OASIS I466 Lake 12 photograph. "
        "Scores and scatter points remain conceptual, not empirical measurements.")
    defs = el(root, "defs")
    for name, colors in [("panel-fill", ["#fbfcfd", "#f4f6f8"]),
                         ("model-fill", ["#fafdff", "#f1f7fe"]),
                         ("table-head", ["#f6f8fa", "#e7edf2"]),
                         ("token-fill", ["#f1f6fb", "#dce6ef"]),
                         ("continuation-fill", ["#ecfaff", "#bfddfb"])]:
        grad = el(defs, "linearGradient", id=name, x1="0%", y1="0%", x2="100%", y2="100%")
        for offset, c in zip(["0%", "100%"], colors):
            el(grad, "stop", offset=offset, stop_color=c)
    for name, d in [("arrow", "M 1 1 L 6 4 L 1 7"),
                    ("arrow-start", "M 6 1 L 1 4 L 6 7")]:
        marker = el(defs, "marker", id=name, markerWidth=8, markerHeight=8,
                    refX=6 if name == "arrow" else 1, refY=4, orient="auto", markerUnits="userSpaceOnUse")
        path(marker, d, sw=1.6, stroke_linecap="round", stroke_linejoin="round")
    background = group(root, "background-and-dividers", True)
    rect(background, 0, 0, WIDTH, HEIGHT, stroke="none", r=0)
    line(background, 20, 461, 1516, 461, "#e0e2e5", 5)
    line(background, 20, 744, 1516, 744, "#e0e2e5", 5)

    a = group(root, "a-the-scored-event", True)
    text(a, 26, 47, "a", 51, bold=True)
    text(a, 80, 47, "The scored event", 36, bold=True)

    inputs = group(a, "supported-input-types")
    rect(inputs, 25, 67, 290, 107, fill="url(#panel-fill)")
    text(inputs, 37, 88, "Supported input types (single input " + it("x") + ")", 15.5, rich=True)
    icon(inputs, "file-text", 43, 104, 37, stroke=1.45)
    icon(inputs, "image", 103, 104, 39, stroke=1.5)
    combined = group(inputs, "icon-image-and-text")
    rect(combined, 173, 109, 49, 30, fill="none", stroke=INKCOLOR, sw=1.8, r=0)
    icon(combined, "image", 175, 114, 23, stroke=1.7)
    for yy in [117, 123, 129]:
        line(combined, 201, yy, 217, yy, sw=1.6)
    icon(inputs, "film", 253, 104, 40, stroke=1.6)
    path(inputs, "M 270 116 L 270 129 L 279 122.5 Z", fill=INKCOLOR, sw=0)
    for xx, label in [(62, "Text"), (123, "Image"), (198, "Image + text"), (274, "Video")]:
        text(inputs, xx, 160, label, 14, anchor="middle")

    material = group(a, "material-x-real-OASIS-stimulus")
    rect(material, 25, 193, 273, 233)
    text(material, 38, 215, "Material " + it("x"), 19, bold=True, rich=True)
    photo(material, 38, 224, 245, 154, "OASIS-stimulus-in-scored-event")
    text(material, 38, 397, "OASIS · Lake 12 (I466)", 15.5)
    text(material, 38, 415, "Dataset example; scores below are symbolic.", 12.5, italic=True, color=MUTED)
    arrow(a, 298, 290, 333, 290)

    wording = group(a, "fixed-wording-and-declared-relation")
    rect(wording, 335, 67, 259, 348, fill="url(#panel-fill)")
    text(wording, 351, 90, "Declared relation:", 15, bold=True)
    text(wording, 351, 111, "emotion evocation", 15, bold=True)
    for n, yy, hh in [(1, 129, 82), (2, 223, 61), (3, 296, 61)]:
        rect(wording, 351, yy, 217, hh, stroke="#cbd6e1", r=2)
        text(wording, 361, yy + 20, "Fixed wording " + m("b", str(n)), 15, bold=True, rich=True)
    text(wording, 361, 175, "“The feeling this photograph", 17, italic=True)
    text(wording, 361, 195, "evokes in a viewer is:”", 17, italic=True)
    text(wording, 363, 269, "…", 18)
    text(wording, 363, 342, "…", 18)
    text(wording, 351, 380, "Same relation and language;", 16, italic=True, color=MUTED)
    text(wording, 351, 399, "fixed before scoring.", 16, italic=True, color=MUTED)
    path(wording, "M 575 170 L 582 170 Q 588 170 588 176 L 588 237 Q 588 245 595 245 "
         "Q 588 245 588 253 L 588 317 Q 588 324 582 324 L 575 324", sw=1.5)
    text(a, 660, 200, "Prefix", 18, bold=True, anchor="middle")
    text(a, 660, 224, it("p") + " = " + m("T", "M") + "(" + it("P") + "(" + it("x, b") + "))", 19, anchor="middle", rich=True)
    arrow(a, 595, 245, 725, 245, 2)

    model = group(a, "frozen-model-and-token-scoring")
    rect(model, 734, 92, 442, 249, fill="url(#model-fill)", stroke=BLUE, sw=2.3, r=5)
    text(model, 955, 120, "Frozen multimodal generative model", 24, bold=True, color="#004bb9", anchor="middle")
    text(model, 960, 144, "Qwen3.5-4B", 19, bold=True, color="#004bb9", anchor="middle")
    icon(model, "lock-keyhole", 863, 147, 30, color="#0052bf", stroke=2)
    text(model, 900, 173, "Model weights fixed", 18, italic=True)
    text(model, 853, 217, "Context tokens (from prefix " + it("p") + ")", 15, anchor="middle", rich=True)
    text(model, 1064, 217, "Continuation tokens (" + m("c", "i,t") + ")", 15, anchor="middle", rich=True)
    path(model, "M 765 234 L 765 232 Q 765 230 770 230 L 849 230 Q 853 230 853 226 Q 853 230 857 230 L 936 230 Q 941 230 941 234", sw=1)
    path(model, "M 980 234 L 980 232 Q 980 230 985 230 L 1059 230 Q 1063 230 1063 226 Q 1063 230 1067 230 L 1141 230 Q 1146 230 1146 234", sw=1)
    for xx, token in [(757, m("t", "1")), (805, m("t", "2")), (901, m("t", "n"))]:
        rect(model, xx, 242, 48, 48, fill="url(#token-fill)", stroke="#687d96", r=0)
        text(model, xx+24, 271, token, 21, anchor="middle", rich=True)
    text(model, 877, 271, "⋯", 25, anchor="middle")
    line(model, 957, 220, 957, 304, "#34516f", 1.4, stroke_dasharray="7 4")
    for xx, token in [(967, m("c", "i,1")), (1015, m("c", "i,2")), (1110, m("c", "i,Lᵢ"))]:
        rect(model, xx, 242, 48, 48, fill="url(#continuation-fill)", stroke="#0872ec", r=0)
        text(model, xx+24, 271, token, 21, anchor="middle", rich=True)
    text(model, 1087, 271, "⋯", 25, anchor="middle")
    text(model, 955, 327, "Score the exact continuation", 19, bold=True, color="#003278", anchor="middle")
    arrow(a, 1186, 245, 1229, 245, 2)

    targets = group(a, "target-continuations")
    rect(targets, 1239, 87, 273, 263, fill="url(#panel-fill)", stroke="#ced9e5")
    text(targets, 1256, 117, "Target continuation " + m("a", "i"), 19, bold=True, rich=True)
    for yy, value in [(136, "pleasant"), (188, "unpleasant"), (240, "aroused"), (292, "calm")]:
        rect(targets, 1256, yy, 129, 40, stroke=BLUE, sw=1.4, r=7)
        text(targets, 1320.5, yy+26, value, 20, anchor="middle")
    brace(targets, 1401, 136, 195)
    for yy, value in [(220,"One anchor"), (238,"per scored"), (256,"event")]:
        text(targets, 1429, yy, value, 15.5)

    eq = group(a, "exact-continuation-log-likelihood-formula")
    lhs = score() + " = log " + m("p", "M") + "(" + m("c", "i") + " | " + it("p") + ") ="
    text(eq, 704, 408, lhs, 28, rich=True)
    text(eq, 1008, 412, "∑", 46, anchor="middle")
    text(eq, 1008, 370, m("L", "i"), 20, anchor="middle", rich=True)
    text(eq, 1008, 435, it("t") + "=1", 18, anchor="middle", rich=True)
    rhs = "log " + m("p", "M") + "(" + m("c", "i,t") + " | " + it("p") + ", " + m("c", "i,<t") + ")."
    text(eq, 1035, 408, rhs, 27, rich=True)
    text(a, 1317, 406, "Natural log probability; nats.", 16, italic=True, color=MUTED)

    b = group(root, "b-from-scores-to-named-coordinates", True)
    text(b, 26, 505, "b", 51, bold=True)
    text(b, 80, 504, "From scores to named coordinates", 35, bold=True)
    scores = group(b, "scores-table")
    rect(scores, 78, 517, 272, 210, fill="#fcfdff")
    text(scores, 92, 540, "Scores for one material " + it("x"), 18, bold=True, rich=True)
    table(scores, 92, 556, [128, 118], ["Anchor " + m("a", "i"), "Score (nats)"],
          [(name, score(str(i))) for i, name in enumerate(["pleasant", "unpleasant", "aroused", "calm"], 1)] + [("…", "…")], row_h=27)
    text(b, 444, 594, "Aggregate across", 18, anchor="middle")
    text(b, 444, 614, "wordings " + it("b"), 18, anchor="middle", rich=True)
    arrow(b, 363, 626, 529, 626)
    text(b, 444, 650, "(e.g., mean)", 15.5, anchor="middle")

    coords = group(b, "named-coordinates-table")
    rect(coords, 551, 517, 315, 210, fill="#fcfdff")
    text(coords, 565, 540, "Named coordinates for material " + it("x"), 18, bold=True, color="#08264a", rich=True)
    table(coords, 565, 556, [183, 103], ["Coordinate (construct)", "Value"],
          [(name, score(str(i), b=False, bar=True)) for i, name in enumerate(["Pleasantness", "Unpleasantness", "Arousal", "Calmness"], 1)] + [("…", "…")], row_h=27)
    text(b, 998, 594, "Coordinates for many", 18, anchor="middle")
    text(b, 998, 614, "materials form a dataset", 18, anchor="middle")
    arrow(b, 878, 626, 1119, 626)
    matrix = it("X") + " → [" + ", ".join(score(str(i), b=False, bar=True) for i in range(1, 5)) + ", …]"
    text(b, 998, 653, matrix, 16.5, anchor="middle", rich=True)

    space = group(b, "conceptual-measurement-space")
    rect(space, 1133, 517, 379, 210, fill="url(#panel-fill)")
    text(space, 1147, 540, "Measurement space", 18, bold=True, color=DARKBLUE)
    # Fixed conceptual points reproduce the reference; these are not real ratings.
    points = [(1301,568),(1354,572),(1391,569),(1277,588),(1311,588),
              (1345,587),(1372,586),(1398,590),(1258,606),(1277,600),
              (1295,607),(1344,607),(1362,601),(1385,602),(1301,633),
              (1347,636),(1364,629),(1258,648),(1285,648),(1301,655),
              (1339,653),(1359,654),(1377,645),(1400,650),(1257,674),
              (1277,669),(1308,674),(1337,659),(1383,667)]
    for i, (xx, yy) in enumerate(points):
        el(space, "circle", id=f"schematic-point-{i+1}", cx=xx,
           cy=round(633 + (yy - 618) * .82, 2), r=2.35,
           fill="#8a98a8", opacity=.58 if i%3 else .8)
    arrow(space, 1246, 633, 1408, 633, double=True)
    arrow(space, 1325, 696, 1325, 570, double=True)
    for xx, yy, value, anch in [(1336,570,"Arousal","start"), (1336,587,"(higher)","start"),
        (1193,629,"Unpleasantness","middle"), (1193,646,"(higher)","middle"),
        (1456,629,"Pleasantness","middle"), (1456,646,"(higher)","middle"),
        (1340,697,"Calmness","start"), (1340,714,"(higher)","start")]:
        text(space, xx, yy, value, 15, anchor=anch)

    c = group(root, "c-two-validation-routes", True)
    text(c, 26, 785, "c", 51, bold=True)
    text(c, 80, 783, "Two validation routes", 36, bold=True)
    rect(c, 25, 798, 823, 233, fill="#fff")
    rect(c, 886, 798, 626, 233, fill="#fff")
    line(c, 866, 798, 866, 1031, "#929ba7", 1.3, stroke_dasharray="8 5")
    text(c, 38, 819, "1. Direct measurement (no training)", 19, bold=True, color="#081e3a")
    text(c, 899, 819, "2. Trained prediction (downstream model)", 19, bold=True, color="#081e3a")

    direct = group(c, "direct-measurement-route")
    rect(direct, 41, 842, 132, 140, fill="#fcfdff")
    text(direct, 54, 865, "New material " + it("x"), 15, bold=True, color="#08264a", rich=True)
    photo(direct, 54, 887, 84, 68, "OASIS-stimulus-in-validation")
    text(direct, 146, 927, "…", 18)
    arrow(direct, 182, 912, 198, 912)
    rect(direct, 207, 842, 184, 140)
    text(direct, 220, 865, "Same fixed wordings " + it("b"), 15.5, rich=True)
    text(direct, 220, 885, "and anchors " + m("a", "i"), 15.5, rich=True)
    text(direct, 220, 918, "Score with frozen model", 15.5)
    text(direct, 299, 954, score() + " → " + score(b=False, bar=True), 18, rich=True, anchor="middle")
    arrow(direct, 400, 912, 416, 912)
    rect(direct, 425, 842, 199, 140, fill="url(#model-fill)", stroke="#bddcff")
    text(direct, 438, 865, "Compare with external", 15, bold=True, color="#08264a")
    text(direct, 438, 885, "human criterion", 15, bold=True, color="#08264a")
    text(direct, 438, 914, "e.g., self-reports,", 14.5, italic=True, color="#24364e")
    text(direct, 438, 934, "behavioural measures,", 14.5, italic=True, color="#24364e")
    text(direct, 438, 954, "or expert ratings", 14.5, italic=True, color="#24364e")
    arrow(direct, 633, 912, 649, 912)
    text(direct, 662, 865, "Evaluate measurement", 14.5, bold=True, color="#11243c")
    text(direct, 662, 885, "quality", 14.5, bold=True, color="#11243c")
    for yy, label in [(908,"Correlation"),(928,"Reliability"),(948,"Construct validity"),(968,"etc.")]:
        text(direct, 662, yy, "•", 14.5)
        text(direct, 677, yy, label, 15)

    trained = group(c, "trained-prediction-route")
    rect(trained, 900, 842, 226, 172)
    text(trained, 913, 865, "Coordinates for many materials", 14.5, bold=True, color="#08264a")
    tr = []
    for i in range(1,4):
        xj = m("x", str(i))
        tr.append((xj, "[" + m("s̄", "1") + "(" + xj + "), …]"))
    tr.append(("…", "…"))
    table(trained, 913, 878, [75,125], ["Material","Coordinates"], tr, row_h=24, fs=14, header_fs=14)
    text(trained, 1203, 872, "Train a prediction", 15, bold=True, anchor="middle")
    text(trained, 1203, 892, "model", 15, bold=True, anchor="middle")
    arrow(trained, 1138, 912, 1268, 912)
    text(trained, 1203, 939, "e.g., regression", 14.5, italic=True, color=MUTED, anchor="middle")
    text(trained, 1203, 959, "or classification", 14.5, italic=True, color=MUTED, anchor="middle")
    rect(trained, 1280, 842, 216, 88, fill="url(#model-fill)", stroke="#bddcff")
    text(trained, 1293, 865, "Predict external criterion", 14.5, bold=True, color="#003278")
    text(trained, 1293, 892, it("Ŷ") + " (e.g., trait scores,", 14.5, rich=True)
    text(trained, 1293, 912, "behavioural outcomes)", 14.5)
    text(trained, 1293, 953, "Evaluate predictive performance", 14.5, bold=True, color="#142238")
    for yy, label in [(975,"Out-of-sample accuracy"),(995,"Generalisation"),(1015,"etc.")]:
        text(trained, 1293, yy, "•", 14.5)
        text(trained, 1309, yy, label, 14.5)
    return root


def to_spec(node):
    return {"tag": node.tag, "attributes": node.attrib, "text": node.text,
            "tail": node.tail, "children": [to_spec(child) for child in node]}


def from_spec(spec):
    node = ET.Element(spec["tag"], spec["attributes"])
    node.text, node.tail = spec["text"], spec["tail"]
    for child in spec["children"]:
        node.append(from_spec(child))
    return node


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-spec", type=Path, help="Render an edited scene JSON instead of rebuilding positions")
    args = parser.parse_args()
    root = from_spec(json.loads(args.render_spec.read_text())["scene"]) if args.render_spec else build()
    if not args.render_spec:
        spec = {"format": "editable-svg-scene-1", "canvas": {"width":WIDTH,"height":HEIGHT},
                "source": "user-provided raster figure", "scene": to_spec(root)}
        (HERE / "figure.scene.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n")
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if value.startswith("@asset:"):
                asset = HERE / value.removeprefix("@asset:")
                data = base64.b64encode(asset.read_bytes()).decode("ascii")
                node.set(key, "data:image/jpeg;base64," + data)
    def indent_svg(node, level=0):
        # Do not indent rich text: XML whitespace changes visible typography.
        if node.tag == f"{{{SVG}}}text" or not len(node):
            return
        node.text = "\n" + "  " * (level+1)
        for child in node:
            indent_svg(child, level+1)
            child.tail = "\n" + "  " * (level+1)
        node[-1].tail = "\n" + "  " * level
    indent_svg(root)
    path_out = HERE / "OmniAnchor_editable.svg"
    ET.ElementTree(root).write(path_out, encoding="utf-8", xml_declaration=True)
    print(path_out)


if __name__ == "__main__":
    main()
