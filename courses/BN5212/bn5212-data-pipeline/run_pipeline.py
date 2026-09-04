#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Build a validated MIMIC-IV + MIMIC-CXR unified index")
    parser.add_argument("--config", default="config/default.json")
    parser.add_argument("--download-manifest", help="Private manifest passed to download_data.py before processing")
    parser.add_argument("--test-loader", action="store_true", help="Load one batch from each split after processing")
    args = parser.parse_args()
    if args.download_manifest:
        from src.data.download import download_manifest
        download_manifest(args.download_manifest)
    from src.data.pipeline import build
    output, summary = build(args.config)
    print(json.dumps({"output": str(output), **summary["counts"]}, indent=2))
    if args.test_loader:
        from test_dataloader import check_run
        check_run(output)


if __name__ == "__main__":
    main()
