#!/usr/bin/env python3
"""Check report word count and list unresolved measurement placeholders."""

import re
from pathlib import Path

path = Path(__file__).with_name("final_report_template.md")
text = path.read_text(encoding="utf-8")
body = re.split(r"^## References", text, maxsplit=1, flags=re.MULTILINE)[0]
body = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", body)
body = re.sub(r"\[[^\]]*\]\([^)]*\)", " ", body)
words = re.findall(r"\b[\w₹]+(?:[.’'-][\w]+)*\b", body)
placeholders = re.findall(r"\[(?:STUDENT|MEASURE):[^\]]*\]", text)
print(f"Draft body word count (tables included): {len(words)}; target 2,500–3,000.")
print(f"Unresolved student/measurement placeholders: {len(placeholders)}")
for item in placeholders:
    print(" -", item)
if placeholders:
    raise SystemExit("Report is intentionally not submission-ready until placeholders are completed.")
if not 2500 <= len(words) <= 3000:
    raise SystemExit("Report body is outside the required word-count range.")
