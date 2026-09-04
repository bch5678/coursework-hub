#!/usr/bin/env python3
import argparse
from src.data.download import download_manifest

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch authorized files listed in a private manifest")
    parser.add_argument("--manifest", required=True)
    download_manifest(parser.parse_args().manifest)
