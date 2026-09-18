# Figure scripts

Main-text Figures 2-7 (`fig2_text.py` ... `fig7_structure.py`) and Supplementary Figures 1-6 (`sfig1_coverage.py` ... `sfig6_video.py`) for the OmniAnchor manuscript. `ncstyle.py` holds the visual system. `fig1_workflow_alt.py` is an alternative schematic; the submitted Figure 1 is the editable vector design in `../figure1-editable/`.

Every script reads the frozen run artifacts under `runs/` (released with the raw scores on acceptance) and the SI file `si.tex` for the two tables it quotes; `ROOT` points at the repository root. Figures use Times New Roman from the system font files; without it matplotlib falls back to its default serif face.
