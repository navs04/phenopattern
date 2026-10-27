#!/usr/bin/env python
"""Download HPO files (once) and build ml/data/processed/*.

Usage:
  python scripts/prepare_data.py                      # download + build
  python scripts/prepare_data.py --hp-obo X --hpoa Y  # use files you already have

Download URLs change occasionally. If one fails, get the files manually from
https://hpo.jax.org/data/ontology and https://hpo.jax.org/data/annotations
and pass them with --hp-obo / --hpoa. Raw files are git-ignored.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from phenopattern.prepare import build_processed  # noqa: E402

# Tried in order; first success wins. VERIFY these on the HPO site if they fail.
OBO_URLS = ["https://purl.obolibrary.org/obo/hp.obo"]
HPOA_URLS = [
    "https://purl.obolibrary.org/obo/hp/hpoa/phenotype.hpoa",
    "https://github.com/obophenotype/human-phenotype-ontology/releases/latest/download/phenotype.hpoa",
]


def download(urls: list[str], dest: Path) -> None:
    last: Exception | None = None
    for url in urls:
        try:
            print(f"Downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as r:  # follows redirects
                dest.write_bytes(r.read())
            return
        except Exception as e:  # noqa: BLE001 - we want to try the next URL
            print(f"  failed: {e}")
            last = e
    raise SystemExit(f"Could not download {dest.name}: {last}. Download it manually (see --help).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out-dir", type=Path, default=ROOT / "data" / "processed")
    ap.add_argument("--hp-obo", type=Path, help="use this local hp.obo instead of downloading")
    ap.add_argument("--hpoa", type=Path, help="use this local phenotype.hpoa instead of downloading")
    args = ap.parse_args()

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    obo = args.hp_obo or args.raw_dir / "hp.obo"
    hpoa = args.hpoa or args.raw_dir / "phenotype.hpoa"
    if not obo.exists():
        download(OBO_URLS, obo)
    if not hpoa.exists():
        download(HPOA_URLS, hpoa)

    meta = build_processed(obo, hpoa, args.out_dir)
    print("\nBuilt processed data:")
    for k in ("hpo_version", "n_terms", "n_phenotype_terms", "n_diseases"):
        print(f"  {k}: {meta[k]}")
    print(f"  hpoa parse stats: {meta['hpoa_parse_stats']}")
    print(f"  written to {args.out_dir}")


if __name__ == "__main__":
    main()
