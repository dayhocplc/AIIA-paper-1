#!/usr/bin/env python3
"""Build the reference list and renumber in-text citations.

Metadata comes from CrossRef (DOI) and the arXiv API, not from memory. The
manuscript cites in Elsevier numbered style [n], numbered by order of first
appearance; a BibTeX file is emitted alongside so the style can be switched in
one step if the journal requires author-date.
"""
import json, re, os

SP = '/tmp/claude-1000/-home-thanhtran-Gia-mia/34d762b6-1451-4926-8bc7-cfc97a0f67f2/scratchpad/'
MS = '<GIAMIA_ROOT>/ov/paper/manuscript.md'
BIB = '<GIAMIA_ROOT>/ov/paper/references.bib'
raw = json.load(open(SP + 'refs_raw.json'))

# Markers placed in the manuscript text -> reference key. Order here is the order
# of first appearance, which is also the numbering order.
ORDER = ['review2025', 'papa2026', 'jin2022', 'rf100vl', 'aerialovd', 'sam3',
         'vfmplant', 'negbench', 'borji22', 'borji20', 'labelconv', 'openimages',
         'sensors2020', 'spray2023', 'woebbecke95', 'meyer2008', 'otsu79',
         'uavvlm', 'owlv2', 'gdino', 'yoloworld', 'qwen25vl', 'rtmdet', 'coco']

# Two entries have no resolvable DOI record; typed from the verified source page
# and flagged for manual check before submission.
MANUAL = {
 'jin2022': {'authors': ['Jin, X.', 'Che, J.', 'Chen, Y.'],
             'title': 'Weed identification using deep learning and image processing in vegetable plantation',
             'container': 'IEEE Access', 'volume': '9', 'page': '10940-10950',
             'year': 2021, 'doi': '10.1109/ACCESS.2021.3050296',
             'note': 'CHECK: cited via the 2025 review as "Jin et al. 2022c"; '
                     'confirm which Jin et al. paper the review means'},
 'uavvlm':  {'authors': ['[authors]'],
             'title': 'Vision-language models for zero-shot weed detection and visual '
                      'reasoning in UAV-based precision agriculture',
             'container': 'PMC12894358', 'year': 2026, 'doi': None,
             'note': 'CHECK: full author list and journal not yet resolved; '
                     'read via PubMed Central'},
}


def clean_author(a):
    """CrossRef records for older papers sometimes carry the whole name in
    `family` with `given` empty, which produced entries like 'D. M. Woebbecke, .'.
    Normalise both shapes to 'Family, I.'."""
    a = a.strip().rstrip(',').strip()
    a = re.sub(r',\s*\.?\s*$', '', a)
    if ',' in a:
        fam, giv = [x.strip() for x in a.split(',', 1)]
        if giv:
            # re-add the period an initial loses to stripping: 'H' -> 'H.'
            giv = ' '.join(g if g.endswith('.') or len(g) > 1 else g + '.'
                           for g in giv.split())
            return f'{fam}, {giv}'
        a = fam
    parts = a.split()
    if len(parts) > 1:
        inits = ' '.join(p[0].upper() + '.' for p in parts[:-1] if p and p[0].isalpha())
        return f'{parts[-1]}, {inits}'.strip().rstrip(',')
    return a


def fmt(k):
    r = MANUAL.get(k) or raw.get(k, {})
    au = [clean_author(x) for x in (r.get('authors') or []) if x and x.strip(' .,')]
    if len(au) > 6:
        aus = ', '.join(au[:6]) + ', et al.'
    else:
        aus = ', '.join(au)
    bits = [aus, r.get('title', '').rstrip('.') + '.']
    cont = r.get('container') or ''
    tail = cont
    if r.get('volume'):
        tail += f" {r['volume']}"
        if r.get('issue'):
            tail += f"({r['issue']})"
    if r.get('article'):
        tail += f", {r['article']}"
    elif r.get('page'):
        tail += f", {r['page']}"
    if r.get('year'):
        tail += f" ({r['year']})"
    bits.append(tail + '.')
    if r.get('doi'):
        bits.append(f"https://doi.org/{r['doi']}")
    line = ' '.join(x for x in bits if x.strip(' .'))
    return line, r.get('note')


lines, notes = [], []
for i, k in enumerate(ORDER, 1):
    txt, note = fmt(k)
    lines.append(f'[{i}] {txt}')
    if note:
        notes.append(f'[{i}] {note}')

ref_md = '## References\n\n' + '\n\n'.join(lines) + '\n'
if notes:
    ref_md += ('\n*Entries needing manual verification before submission:*\n\n'
               + '\n'.join(f'- {n}' for n in notes) + '\n')

s = open(MS).read()
if '## References' in s:
    s = s[:s.index('## References')] + ref_md
else:
    s = s.rstrip() + '\n\n---\n\n' + ref_md
open(MS, 'w').write(s)

# BibTeX so the style can be changed in one step
def bib(k):
    r = MANUAL.get(k) or raw.get(k, {})
    au = ' and '.join(r.get('authors') or [])
    typ = 'misc' if str(r.get('container', '')).startswith('arXiv') else 'article'
    out = [f'@{typ}{{{k},']
    out.append(f'  author = {{{au}}},')
    out.append(f'  title = {{{r.get("title","")}}},')
    if r.get('container'):
        out.append(f'  journal = {{{r["container"]}}},')
    for f, key in (('volume', 'volume'), ('number', 'issue'), ('pages', 'page')):
        if r.get(key):
            out.append(f'  {f} = {{{r[key]}}},')
    if r.get('year'):
        out.append(f'  year = {{{r["year"]}}},')
    if r.get('doi'):
        out.append(f'  doi = {{{r["doi"]}}},')
    out.append('}')
    return '\n'.join(out)


open(BIB, 'w').write('\n\n'.join(bib(k) for k in ORDER) + '\n')
print(f'{len(ORDER)} references written to the manuscript')
print(f'BibTeX -> {BIB}')
print(f'{len(notes)} entry(ies) flagged for manual verification')
for n in notes:
    print('  ' + n)
