#!/usr/bin/env python3
"""
Local Citations Utility for TIC

Features:
- Add citations by DOI or URL (Crossref, PubMed, arXiv supported)
- Enrich metadata (authors, year, title, journal, etc.)
- Generate BibTeX, RIS, APA outputs in citations/output/

Usage examples:
  python3 citations/manage_citations.py add --doi 10.1016/j.neuron.2024.11.008 --tag TIC_v2.1_update
  python3 citations/manage_citations.py add --url https://arxiv.org/abs/1801.03924 --tag TIC_v2.1_update
  python3 citations/manage_citations.py enrich
  python3 citations/manage_citations.py generate

This tool prefers the 'requests' package; if unavailable, it falls back to urllib.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Optional requests import with fallback
try:
    import requests  # type: ignore
    def http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        return r.text
except Exception:
    import urllib.request
    import urllib.parse
    def http_get(url: str, params: Optional[dict] = None, headers: Optional[dict] = None):
        if params:
            query = urllib.parse.urlencode(params)
            sep = '&' if ('?' in url) else '?'
            url = f"{url}{sep}{query}"
        req = urllib.request.Request(url, headers=headers or {})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode('utf-8', errors='replace')


BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / 'citations.json'
OUT_DIR = BASE_DIR / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_url(url: str) -> str:
    url = url.strip()
    url = re.sub(r'^https?://', '', url)
    url = url.rstrip('/')
    return url.lower()


def extract_pmid(url: str) -> Optional[str]:
    pats = [r'pubmed\.ncbi\.nlm\.nih\.gov/(\d+)', r'ncbi\.nlm\.nih\.gov/pubmed/(\d+)', r'pmid[:/](\d+)']
    for p in pats:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def extract_doi(text: str) -> Optional[str]:
    pats = [r'doi\.org/(10\.\d+/[^\s]+)', r'doi[:/](10\.\d+/[^\s]+)', r'(10\.\d{4,}/[^\s]+)']
    for p in pats:
        m = re.search(p, text)
        if m:
            return m.group(1).rstrip('.')
    return None


def extract_arxiv_id(url: str) -> Optional[str]:
    pats = [r'arxiv\.org/abs/(\d+\.\d+)', r'arxiv\.org/pdf/(\d+\.\d+)', r'arxiv[:/](\d+\.\d+)']
    for p in pats:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None


def fetch_crossref(doi: str) -> Optional[Dict]:
    url = f"https://api.crossref.org/works/{doi}"
    try:
        txt = http_get(url, headers={'User-Agent': 'TIC-Citations/1.0'})
        import json as _json
        data = _json.loads(txt)
        msg = data.get('message') or {}
        authors = []
        for a in msg.get('author', []) or []:
            given = a.get('given', '')
            family = a.get('family', '')
            authors.append(f"{family}, {given}".strip(', '))
        # year
        date_parts = (msg.get('published-print', {}) or {}).get('date-parts', [[]])
        if not date_parts or not date_parts[0]:
            date_parts = (msg.get('published-online', {}) or {}).get('date-parts', [[]])
        year = (date_parts[0][0] if (date_parts and date_parts[0]) else None)
        return {
            'title': (msg.get('title') or [''])[0],
            'authors': authors or ['[Unknown]'],
            'journal': (msg.get('container-title') or [''])[0],
            'year': year,
            'volume': msg.get('volume'),
            'issue': msg.get('issue'),
            'pages': msg.get('page'),
            'publisher': msg.get('publisher'),
            'type': (msg.get('type') or 'journal-article').replace('-', '_'),
        }
    except Exception:
        return None


def fetch_pubmed(pmid: str) -> Optional[Dict]:
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    try:
        txt = http_get(base, params={'db': 'pubmed', 'id': pmid, 'retmode': 'json'})
        import json as _json
        data = _json.loads(txt)
        article = (data.get('result') or {}).get(pmid)
        if not article:
            return None
        authors = [a.get('name') for a in article.get('authors', []) if a.get('name')]
        year = None
        if article.get('pubdate'):
            try:
                year = int(article['pubdate'].split()[0])
            except Exception:
                year = None
        return {
            'title': article.get('title', ''),
            'authors': authors or ['[Unknown]'],
            'journal': article.get('source', ''),
            'year': year,
            'volume': article.get('volume'),
            'issue': article.get('issue'),
            'pages': article.get('pages'),
            'type': 'journal_article'
        }
    except Exception:
        return None


def fetch_arxiv(arxiv_id: str) -> Optional[Dict]:
    api = "http://export.arxiv.org/api/query"
    try:
        txt = http_get(api, params={'id_list': arxiv_id, 'max_results': 1})
        # crude XML scraping
        title = re.search(r'<title>(.*?)</title>', txt, re.DOTALL)
        title = title.group(1).strip() if title else ''
        authors = re.findall(r'<name>(.*?)</name>', txt)
        year = None
        m = re.search(r'<published>(\d{4})', txt)
        if m:
            year = int(m.group(1))
        abstr = re.search(r'<summary>(.*?)</summary>', txt, re.DOTALL)
        abstract = (abstr.group(1).strip() if abstr else '')
        return {
            'title': title,
            'authors': authors or ['[Unknown]'],
            'journal': 'arXiv',
            'year': year,
            'type': 'preprint',
            'abstract': abstract[:500] if abstract else None,
        }
    except Exception:
        return None


def make_key(entry: Dict) -> str:
    authors = entry.get('authors') or ['unknown']
    first = authors[0]
    last = first.split(',')[0] if ',' in first else (first.split()[-1] if first else 'unknown')
    last = re.sub(r'[^A-Za-z]', '', last).lower() or 'unknown'
    year = entry.get('year') or 'nd'
    word = 'unknown'
    if entry.get('title'):
        words = re.findall(r'\b\w+\b', entry['title'])
        for w in words:
            if len(w) > 4:
                word = w.lower()[:5]
                break
    return f"{last}{year}{word}"


def clean(text: Optional[str]) -> str:
    return '' if text is None else str(text)


def to_bibtex(bib: List[Dict]) -> str:
    parts = []
    type_map = {
        'journal_article': 'article',
        'journal-article': 'article',
        'book': 'book',
        'book_chapter': 'inbook',
        'book_section': 'inbook',
        'preprint': 'misc',
        'conference': 'inproceedings',
        'thesis': 'phdthesis',
    }
    for e in bib:
        if not e.get('title') or e.get('title') == '[Pending extraction]':
            continue
        key = e.get('id') or make_key(e)
        typ = type_map.get(e.get('type', 'misc'), 'misc')
        s = [f"@{typ}{{{key},"]
        if e.get('title'): s.append(f"  title = {{{clean(e['title'])}}},")
        if e.get('authors'): s.append(f"  author = {{{' and '.join(e['authors'][:10])}}},")
        if e.get('year'): s.append(f"  year = {{{e['year']}}},")
        if e.get('journal'): s.append(f"  journal = {{{clean(e['journal'])}}},")
        if e.get('volume'): s.append(f"  volume = {{{e['volume']}}},")
        if e.get('issue'): s.append(f"  number = {{{e['issue']}}},")
        if e.get('pages'): s.append(f"  pages = {{{e['pages']}}},")
        if e.get('doi'): s.append(f"  doi = {{{e['doi']}}},")
        if e.get('url'): s.append(f"  url = {{{e['url']}}},")
        if e.get('publisher'): s.append(f"  publisher = {{{clean(e['publisher'])}}},")
        s.append("}\n")
        parts.append('\n'.join(s))
    return '\n'.join(parts)


def to_ris(bib: List[Dict]) -> str:
    parts = []
    type_map = {
        'journal_article': 'JOUR',
        'journal-article': 'JOUR',
        'book': 'BOOK',
        'book_chapter': 'CHAP',
        'book_section': 'CHAP',
        'preprint': 'UNPB',
        'conference': 'CONF',
        'thesis': 'THES'
    }
    for e in bib:
        if not e.get('title') or e.get('title') == '[Pending extraction]':
            continue
        typ = type_map.get(e.get('type', 'misc'), 'GEN')
        s = [f"TY  - {typ}"]
        s.append(f"TI  - {clean(e.get('title'))}")
        for a in (e.get('authors') or [])[:10]:
            s.append(f"AU  - {a}")
        if e.get('year'): s.append(f"PY  - {e['year']}")
        if e.get('journal'): s.append(f"JO  - {e['journal']}")
        if e.get('volume'): s.append(f"VL  - {e['volume']}")
        if e.get('issue'): s.append(f"IS  - {e['issue']}")
        if e.get('pages'): s.append(f"SP  - {e['pages']}")
        if e.get('doi'): s.append(f"DO  - {e['doi']}")
        if e.get('url'): s.append(f"UR  - {e['url']}")
        if e.get('publisher'): s.append(f"PB  - {e['publisher']}")
        for tag in (e.get('tags') or [])[:10]:
            s.append(f"KW  - {tag}")
        s.append("ER  - ")
        parts.append('\n'.join(s) + '\n')
    return '\n'.join(parts)


def to_apa(bib: List[Dict]) -> str:
    out = ["# References (APA Format)", ""]
    def format_authors(auth: List[str]) -> str:
        if not auth: return ''
        if len(auth) == 1: return auth[0]
        if len(auth) == 2: return f"{auth[0]}, & {auth[1]}"
        if len(auth) <= 20: return ', '.join(auth[:-1]) + f", & {auth[-1]}"
        return ', '.join(auth[:19]) + f", ... {auth[-1]}"

    for e in bib:
        if not e.get('title') or e.get('title') == '[Pending extraction]':
            continue
        authors = format_authors(e.get('authors') or []) or '[Author unknown]'
        year = e.get('year') or 'n.d.'
        title = e.get('title') or 'Untitled'
        pub_parts = []
        if e.get('journal'):
            vol = ''
            if e.get('volume'):
                vol = f"*{e['volume']}*"
                if e.get('issue'):
                    vol += f"({e['issue']})"
            if e.get('pages'):
                vol = f"{vol}, {e['pages']}" if vol else e['pages']
            pub_parts.append(f"*{e['journal']}*, {vol}" if vol else f"*{e['journal']}*")
        elif e.get('publisher'):
            pub_parts.append(e['publisher'])
        if e.get('doi'):
            pub_parts.append(f"https://doi.org/{e['doi']}")
        elif e.get('url'):
            pub_parts.append(e['url'])
        line = f"{authors} ({year}). {title}."
        if pub_parts:
            line += ' ' + '. '.join(pub_parts)
        if not line.endswith('.'): line += '.'
        out.append(line)
        out.append('')
    return '\n'.join(out)


@dataclass
class CitationDB:
    path: Path
    entries: List[Dict] = field(default_factory=list)

    def load(self):
        if self.path.exists():
            self.entries = json.loads(self.path.read_text())
        else:
            self.entries = []

    def save(self):
        self.path.write_text(json.dumps(self.entries, indent=2, ensure_ascii=False))

    def add(self, url: Optional[str], doi: Optional[str], tag: Optional[str]):
        url = url.strip() if url else None
        doi = doi.strip() if doi else None
        norm = normalize_url(url) if url else None
        # dedupe by url or doi
        for e in self.entries:
            if doi and e.get('doi') and e['doi'].lower() == doi.lower():
                if tag and tag not in (e.get('tags') or []):
                    e.setdefault('tags', []).append(tag)
                return e
            if url and e.get('url') and normalize_url(e['url']) == norm:
                if tag and tag not in (e.get('tags') or []):
                    e.setdefault('tags', []).append(tag)
                return e
        # create new
        import hashlib
        base = doi or (url or 'new')
        h = hashlib.md5(base.encode()).hexdigest()[:8]
        entry = {
            'id': f'pending_{h}',
            'url': url,
            'doi': doi,
            'source': 'TIC_local',
            'validation_status': 'pending',
            'authors': [],
            'year': None,
            'title': '[Pending extraction]',
            'journal': None,
            'volume': None,
            'issue': None,
            'pages': None,
            'pmid': None,
            'type': 'unknown',
            'abstract': None,
            'tags': [tag] if tag else []
        }
        self.entries.append(entry)
        return entry

    def enrich_one(self, e: Dict, delay: float = 0.4):
        url = e.get('url') or ''
        meta = None
        # PubMed
        pmid = extract_pmid(url)
        if pmid:
            meta = fetch_pubmed(pmid)
            if meta:
                e['pmid'] = pmid
        # DOI
        if not meta:
            doi = e.get('doi') or extract_doi(url) or None
            if doi:
                m = fetch_crossref(doi)
                if m:
                    e['doi'] = doi
                    meta = m
        # arXiv
        if not meta:
            arx = extract_arxiv_id(url)
            if arx:
                meta = fetch_arxiv(arx)
        # apply
        if meta:
            for k, v in meta.items():
                if v and (k not in e or e[k] in [None, '', '[Pending extraction]']):
                    e[k] = v
            # finalize id
            if str(e.get('id','')).startswith('pending_'):
                e['id'] = make_key(e)
            e['validation_status'] = 'complete'
            e['enriched_at'] = time.strftime('%Y-%m-%dT%H:%M:%S')
        else:
            # partial/pending
            if e.get('title') and e['title'] != '[Pending extraction]':
                e['validation_status'] = 'partial'
            else:
                e['validation_status'] = 'pending'
        time.sleep(delay)


def cmd_add(args):
    db = CitationDB(DB_PATH)
    db.load()
    entry = db.add(url=args.url, doi=args.doi, tag=args.tag)
    db.save()
    print(f"Added/updated entry: {entry.get('id')}  tags={entry.get('tags')}")


def cmd_enrich(args):
    db = CitationDB(DB_PATH)
    db.load()
    targets = db.entries
    if args.tag:
        targets = [e for e in targets if args.tag in (e.get('tags') or [])]
    print(f"Enriching {len(targets)} entries{' with tag ' + args.tag if args.tag else ''}...")
    for e in targets:
        db.enrich_one(e, delay=args.delay)
    db.save()
    print("Done.")


def cmd_generate(args):
    db = CitationDB(DB_PATH)
    db.load()
    bib = [e for e in db.entries if e.get('title') and e['title'] != '[Pending extraction]']
    # Outputs
    (OUT_DIR / 'references.bib').write_text(to_bibtex(bib))
    (OUT_DIR / 'references.ris').write_text(to_ris(bib))
    (OUT_DIR / 'apa_references.md').write_text(to_apa(bib))
    print("Generated outputs in citations/output/")


def main():
    ap = argparse.ArgumentParser(description='Local citations utility for TIC')
    sub = ap.add_subparsers(dest='cmd')

    ap_add = sub.add_parser('add', help='Add a citation by DOI or URL')
    ap_add.add_argument('--doi', type=str, default=None)
    ap_add.add_argument('--url', type=str, default=None)
    ap_add.add_argument('--tag', type=str, default=None)
    ap_add.set_defaults(func=cmd_add)

    ap_enrich = sub.add_parser('enrich', help='Enrich metadata via Crossref/PubMed/arXiv')
    ap_enrich.add_argument('--tag', type=str, default=None, help='Only enrich entries with this tag')
    ap_enrich.add_argument('--delay', type=float, default=0.4, help='Delay between calls (s)')
    ap_enrich.set_defaults(func=cmd_enrich)

    ap_gen = sub.add_parser('generate', help='Generate outputs (BibTeX, RIS, APA)')
    ap_gen.set_defaults(func=cmd_generate)

    args = ap.parse_args()
    if not args.cmd:
        ap.print_help()
        return
    args.func(args)


if __name__ == '__main__':
    main()

