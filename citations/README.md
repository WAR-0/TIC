# Local Citations Utility (optional)

This folder provides a lightweight, repo‑local citation helper similar to the IRTP project’s tooling. It lets you:

- Add citations by DOI or URL (Crossref, PubMed, arXiv supported)
- Enrich metadata (authors, year, title, journal)
- Generate BibTeX, RIS, and APA‑style reference outputs
- Tag entries (e.g., `TIC_v2.1_update`) for quick filtering

Outputs live in `citations/output/` and are git‑ignored by default.

## Quick Start

```
# From repo root
python3 citations/manage_citations.py add --doi 10.1016/j.neuron.2024.11.008 --tag TIC_v2.1_update
python3 citations/manage_citations.py add --url https://arxiv.org/abs/1801.03924 --tag TIC_v2.1_update

# Enrich (fetch metadata via Crossref/PubMed/arXiv)
python3 citations/manage_citations.py enrich

# Generate outputs
python3 citations/manage_citations.py generate
# Creates:
#  - citations/output/references.bib
#  - citations/output/references.ris
#  - citations/output/apa_references.md
```

## Notes
- The tool attempts to use `requests` if installed; otherwise it falls back to Python’s stdlib HTTP client.
- API calls are rate‑limited lightly; please avoid bulk hammering Crossref/PubMed.
- Tagging entries (e.g., `--tag TIC_v2.1_update`) helps you track batch imports in Zotero/Mendeley (RIS exports include `KW  - tag`).

