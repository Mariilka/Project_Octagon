"""Полный пересчёт проекта одной командой:  python run_all.py"""
import subprocess
import sys
from pathlib import Path

sys.stdout.reconfigure(errors="replace")  # консоль Windows (cp1251) не падает на символах вне кодировки

SRC = Path(__file__).parent / "src"
for step in ("prepare.py", "features.py", "validate.py", "hypotheses.py", "export_bi.py"):
    print(f"\n=== {step} ===")
    subprocess.run([sys.executable, str(SRC / step)], check=True, stdout=subprocess.DEVNULL if step == "hypotheses.py" else None)
print("\nГотово: результаты в папке outputs/")
