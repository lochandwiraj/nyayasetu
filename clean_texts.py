"""
Run this once to strip IndianKanoon UI boilerplate from the txt files
and produce clean versions ready for setup_data.py.

Usage:
    python clean_texts.py
"""

import re
import os

DATA_DIR = r"C:\Users\dwira\OneDrive\Desktop\nayasetu_data"

FILES = {
    "ipc":     ("ipc.txt",     "ipc_clean.txt"),
    "rti":     ("rti.txt",     "rti_clean.txt"),
    "mgnrega": ("mgnrega.txt", "mgnrega_clean.txt"),
    "dv_act":  ("dv_act.txt",  "dv_act_clean.txt"),
}

# Markers that appear just before the actual legal text begins on IndianKanoon
START_MARKERS = [
    "THE INDIAN PENAL CODE",
    "THE RIGHT TO INFORMATION ACT",
    "THE MAHATMA GANDHI NATIONAL RURAL EMPLOYMENT GUARANTEE ACT",
    "THE PROTECTION OF WOMEN FROM DOMESTIC VIOLENCE ACT",
    # fallback: first occurrence of "ACT" after a section number pattern
]

# Boilerplate lines to drop even inside the legal text
NOISE_PATTERNS = [
    r"^Skip to main content",
    r"^Indian Kanoon",
    r"^Search$",
    r"^Search Indian laws",
    r"^Search laws",
    r"^Main Navigation",
    r"^Free features",
    r"^Premium$",
    r"^Prism AI",
    r"^IKademy",
    r"^Pricing$",
    r"^Login$",
    r"^Legal Document View",
    r"^Tools for analyzing",
    r"^Unlock Advanced",
    r"^Integrated with over",
    r"^designed for legal",
    r"^Doc Gen Hub",
    r"^Counter Argument",
    r"^Case Predict AI",
    r"^Talk with IK Doc",
    r"^Upgrade to Premium",
    r"^\[Cites \d+",
    r"^\[Cited by \d+",
    r"^Union of India",
    r"^UNION OF INDIA",
    r"^Published in Gazette",
    r"^Assented to on",
    r"^Commenced on",
    r"^\[This is the version",
    r"^View Complete Act",
    r"^Central Government Act",
    r"^State Government Act",
    r"^Download",
    r"^Print$",
    r"^Share$",
    r"^Bookmark$",
    r"^Follow$",
    r"^Report$",
    r"^Home\s*$",
    r"^Acts\s*$",
    r"^Judgments\s*$",
]

noise_re = re.compile("|".join(NOISE_PATTERNS), re.IGNORECASE)

def find_start(lines: list[str]) -> int:
    """Return index of first line that looks like real legal text."""
    for i, line in enumerate(lines):
        stripped = line.strip()
        # Look for section numbering like "1." or "Section 1" or act title in caps
        if re.match(r'^(Section\s+\d+|CHAPTER|PART\s+[IVX]+|\d+\.\s+[A-Z])', stripped):
            return i
        for marker in START_MARKERS:
            if marker.upper() in stripped.upper() and len(stripped) < 120:
                return i
    return 0  # fallback: keep everything

def clean(text: str) -> str:
    lines = text.splitlines()
    start = find_start(lines)
    lines = lines[start:]

    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        if noise_re.match(stripped):
            continue
        # Drop very short lines that are clearly UI fragments
        if len(stripped) < 4:
            continue
        cleaned.append(stripped)

    return "\n".join(cleaned)

def main():
    for key, (src, dst) in FILES.items():
        src_path = os.path.join(DATA_DIR, src)
        dst_path = os.path.join(DATA_DIR, dst)

        if not os.path.exists(src_path):
            print(f"SKIP  {src} — not found")
            continue

        with open(src_path, "r", encoding="utf-8", errors="ignore") as f:
            raw = f.read()

        cleaned = clean(raw)

        with open(dst_path, "w", encoding="utf-8") as f:
            f.write(cleaned)

        orig_kb  = len(raw.encode()) // 1024
        clean_kb = len(cleaned.encode()) // 1024
        print(f"OK    {src} → {dst}  ({orig_kb}KB → {clean_kb}KB)")

    print("\nDone. Now update setup_data.py to point at the _clean.txt files,")
    print("then re-run:  python setup_data.py")

if __name__ == "__main__":
    main()
