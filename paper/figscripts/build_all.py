#!/usr/bin/env python3
"""
Build the seven r2 main-text figures from frozen analysis data.

Usage:  python paper/nc/r2/figscripts/build_all.py
Output: paper/nc/r2/figures/fig1-workflow.pdf ... fig7-structure.pdf
        paper/nc/r2/figures/preview/*.png  (300 dpi previews)

Acceptance checks (DESIGN-SPEC.md section 5):
  1. Font embedding: only HelveticaNeue-* faces; no Type 3.
  2. Page width <= 183 mm.
  3. PNG previews at 300 dpi for visual inspection.
"""
from __future__ import annotations
import importlib
import sys
import traceback
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

FIGURES = [
    "fig1_workflow",
    "fig2_text",
    "fig3_languages",
    "fig4_images",
    "fig5_video",
    "fig6_sensitivity",
    "fig7_structure",
]


def main() -> int:
    ok, fail = [], []
    for name in FIGURES:
        print(f"\n{'=' * 60}\nBuilding {name}...\n{'=' * 60}")
        try:
            mod = importlib.import_module(name)
            mod.main()
            ok.append(name)
        except Exception:
            traceback.print_exc()
            fail.append(name)

    print(f"\n{'=' * 60}")
    print(f"Done.  {len(ok)} succeeded, {len(fail)} failed.")
    if fail:
        print(f"  Failed: {', '.join(fail)}")
    print(f"{'=' * 60}")

    fig_dir = SCRIPT_DIR.parent / "figures"
    pdfs = sorted(fig_dir.glob("fig*.pdf"))

    problems = []
    try:
        from pypdf import PdfReader
        print("\n--- Font and size checks ---")
        for pdf in pdfs:
            page = PdfReader(str(pdf)).pages[0]
            w_mm = float(page.mediabox.width) * 25.4 / 72
            h_mm = float(page.mediabox.height) * 25.4 / 72
            fonts = set()
            resources = page.get("/Resources") or {}
            if "/Font" in resources:
                for font_ref in resources["/Font"].values():
                    font_obj = font_ref.get_object()
                    fonts.add((str(font_obj.get("/BaseFont", "unknown")),
                               str(font_obj.get("/Subtype", "unknown"))))
            type3 = any("Type3" in st for _, st in fonts)
            non_helv = [n for n, _ in fonts if "HelveticaNeue" not in n]
            status = []
            if w_mm > 183.5:
                status.append("TOO WIDE")
            if h_mm > 247:
                status.append("TOO TALL")
            if type3:
                status.append("TYPE3")
            if non_helv:
                status.append(f"non-HelveticaNeue: {non_helv}")
            tag = " ".join(status) if status else "OK"
            if status:
                problems.append(f"{pdf.name}: {tag}")
            print(f"  {pdf.name}: {w_mm:.1f} x {h_mm:.1f} mm  [{tag}]")
            for name, subtype in sorted(fonts):
                print(f"    font: {name}  ({subtype})")
    except ImportError:
        print("  (pypdf not available for acceptance checks)")
        problems.append("pypdf missing: font and size checks not run")

    if problems:
        print("\n--- Acceptance problems ---")
        for line in problems:
            print(f"  {line}")

    return 1 if (fail or problems) else 0


if __name__ == "__main__":
    sys.exit(main())
