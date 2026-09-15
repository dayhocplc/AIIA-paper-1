#!/usr/bin/env python3
"""Elsevier caps each Highlight at 85 characters including spaces.

Checks BOTH files that carry the highlights: the submission pack and the
manuscript. An earlier edit fixed one and not the other, and the over-length
bullet reached the .docx.
"""
import re, sys

FILES = {
    'submission_checklist.md': ('## 2. Highlights', '## 3.'),
    'manuscript.md': ('## Highlights', '## Abstract'),
}
BASE = '<GIAMIA_ROOT>/ov/paper/'

bad = 0
seen = {}
for fn, (start, end) in FILES.items():
    txt = open(BASE + fn).read()
    sec = txt.split(start)[1].split(end)[0]
    hs = [m.group(1) for m in re.finditer(r'^\s*\d+\.\s+`(.+)`\s*$', sec, re.M)]
    seen[fn] = hs
    print(f'--- {fn} ---')
    for h in hs:
        n = len(h)
        if n > 85:
            bad += 1
        print(f'{"OK " if n <= 85 else "OVER"} {n:3d}  {h}')

a, b = (seen.get(f) or [] for f in FILES)
if a and b and a != b:
    bad += 1
    print('\nMISMATCH: the two files carry different highlight text')
    for i, (x, y) in enumerate(zip(a, b), 1):
        if x != y:
            print(f'  #{i} checklist: {x}')
            print(f'  #{i} manuscript: {y}')

print(f'\n{bad} problem(s)')
sys.exit(1 if bad else 0)
