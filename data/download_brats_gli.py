"""
Download BraTS-GLI 2024 from HuggingFace.

Usage:
    python data/download_brats_gli.py --output_dir data/raw/BraTS_GLI

Source: ClarkQuinn/BraTS_GLI_PRE  (~17 GB)
Requires: huggingface_hub  (pip install huggingface_hub)
HF token required for gated datasets — set HF_TOKEN env var or pass --token.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def download(output_dir: str, token: str | None = None):
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit(
            "huggingface_hub not installed.\n"
            "Run: pip install huggingface_hub"
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    token = token or os.environ.get("HF_TOKEN")

    print(f"Downloading BraTS-GLI 2024 → {out}")
    print("This is ~17 GB and may take 30–60 minutes on a fast connection.")

    path = snapshot_download(
        repo_id="ClarkQuinn/BraTS_GLI_PRE",
        repo_type="dataset",
        local_dir=str(out),
        token=token,
        ignore_patterns=["*.md", "*.txt"],
    )
    print(f"\nDownload complete: {path}")
    print("Files are ready for BraTSGLIDataset (no preprocessing required — NIfTI volumes).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download BraTS-GLI 2024 from HuggingFace")
    parser.add_argument("--output_dir", type=str, default="data/raw/BraTS_GLI")
    parser.add_argument("--token", type=str, default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()
    download(args.output_dir, args.token)
