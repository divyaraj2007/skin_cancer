"""Assemble the conference paper: copy generated tables/numbers/figures into paper/ and zip it for Overleaf.

    python scripts/build_paper.py            # after `python -m src.analysis` and `python -m src.gradcam`

If a LaTeX toolchain (latexmk or pdflatex+bibtex) is on PATH, the PDF is compiled as well.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
GEN = ROOT / "results" / "paper"
FIGS = ["fig2_confusion_matrix.png", "fig3_roc_curves.png", "fig5_calibration.png",
        "fig7_gradcam.png", "fig8_leakage.png", "fig6_training_curves.png", "fig1_dataset_split.png",
        "fig4_per_class.png"]


def main() -> None:
    (PAPER / "latex").mkdir(parents=True, exist_ok=True)
    (PAPER / "figures").mkdir(parents=True, exist_ok=True)
    for f in (GEN / "latex").glob("*.tex"):
        shutil.copy2(f, PAPER / "latex" / f.name)
    missing = []
    for name in FIGS:
        src = GEN / "figures" / name
        if src.exists():
            shutil.copy2(src, PAPER / "figures" / name)
        else:
            missing.append(name)
    if missing:
        print("Warning: missing figures:", missing)

    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    if "%RESULTS%" in tex or "%ABSTRACT%" in tex:
        print("Warning: main.tex still contains unfilled placeholders")

    zip_path = ROOT / "paper_overleaf.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(PAPER.rglob("*")):
            if f.is_file() and f.suffix in {".tex", ".bib", ".png", ".pdf", ".cls", ".bst"} and f.name != "main.pdf":
                z.write(f, f.relative_to(PAPER))
    print(f"Overleaf bundle: {zip_path}")

    if shutil.which("latexmk"):
        subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"], cwd=PAPER, check=False)
    elif shutil.which("pdflatex") and shutil.which("bibtex"):
        for cmd in (["pdflatex", "-interaction=nonstopmode", "main"], ["bibtex", "main"],
                    ["pdflatex", "-interaction=nonstopmode", "main"], ["pdflatex", "-interaction=nonstopmode", "main"]):
            subprocess.run(cmd, cwd=PAPER, check=False, stdout=subprocess.DEVNULL)
    else:
        print("No LaTeX toolchain found: upload paper_overleaf.zip to Overleaf (New Project -> Upload Project).")
    if (PAPER / "main.pdf").exists():
        print(f"PDF: {PAPER / 'main.pdf'}")
    return None


if __name__ == "__main__":
    sys.exit(main())
