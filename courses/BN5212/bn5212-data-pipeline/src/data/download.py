"""Checksum-verified HTTPS downloads and guarded archive extraction.

Private URLs belong in a user's local manifest. Credentials are read only from
environment variables and are never written to a manifest, log or command line.
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import stat
import tarfile
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

from .io import safe_path, sha256, write_json


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("Download redirected; supply the final authorized HTTPS file URL in your private manifest")


def fetch_file(url: str, destination: Path, expected_hash: str):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password:
        raise ValueError("Use an HTTPS URL without embedded credentials")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", expected_hash):
        raise ValueError("Each download requires the expected SHA-256 from a trusted source")
    expected_hash = expected_hash.lower()
    if destination.exists():
        if sha256(destination) == expected_hash:
            return
        raise ValueError("Existing download has the wrong checksum; choose a fresh destination")
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset and sha256(partial) == expected_hash:
        partial.replace(destination)
        return
    headers = {"User-Agent": "BN5212-data-pipeline/1.0", "Accept-Encoding": "identity"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    username, password = os.environ.get("DATA_HTTP_USERNAME"), os.environ.get("DATA_HTTP_PASSWORD")
    if bool(username) != bool(password):
        raise ValueError("Set both DATA_HTTP_USERNAME and DATA_HTTP_PASSWORD, or neither")
    auth_host = os.environ.get("DATA_HTTP_AUTH_HOST", "").strip().lower()
    if username and not auth_host:
        raise ValueError("Set DATA_HTTP_AUTH_HOST to the exact authorized hostname before using Basic Auth")
    if username and parsed.hostname.lower() == auth_host:
        headers["Authorization"] = "Basic " + base64.b64encode(f"{username}:{password}".encode()).decode()
    opener = urllib.request.build_opener(NoRedirect())
    try:
        with opener.open(urllib.request.Request(url, headers=headers), timeout=120) as response:
            if response.status not in (200, 206):
                raise ValueError("Unexpected HTTP status")
            append = offset > 0 and response.status == 206
            if response.status == 206:
                content_range = response.headers.get("Content-Range", "")
                match = re.fullmatch(r"bytes ([0-9]+)-([0-9]+)/([0-9]+|\*)", content_range)
                if not match or int(match.group(1)) != offset:
                    raise ValueError("Invalid resume response")
            mode = "ab" if append else "wb"
            with partial.open(mode) as stream:
                shutil.copyfileobj(response, stream, 1024 * 1024)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP download failed (status {exc.code}); check access, credentials or final URL") from None
    except urllib.error.URLError:
        raise RuntimeError("Network download failed; rerun to resume the partial file") from None
    if sha256(partial) != expected_hash:
        raise ValueError("Downloaded checksum mismatch; check the expected hash and remove the corrupt .part file before retrying")
    partial.replace(destination)


def extract_archive(archive: Path, destination: Path, expected_hash: str, max_bytes: int):
    marker = destination / ".bn5212_archive.json"
    if destination.exists():
        if marker.is_file() and json.loads(marker.read_text())["sha256"] == expected_hash:
            return
        raise ValueError("Extraction destination already exists without a matching completion marker")
    if max_bytes <= 0:
        raise ValueError("max_extract_bytes must be positive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".extract-", dir=destination.parent))
    total, members, seen = 0, 0, set()

    def target(name, size):
        nonlocal total, members
        total += size; members += 1
        if total > max_bytes or members > 1000000:
            raise ValueError("Archive exceeds extraction budget")
        path = safe_path(staging, name)
        canonical = str(path).casefold() if os.name == "nt" else str(path)
        if canonical in seen:
            raise ValueError("Duplicate archive member path")
        seen.add(canonical)
        return path

    if zipfile.is_zipfile(archive):
        password = os.environ.get("DATA_ARCHIVE_PASSWORD")
        with zipfile.ZipFile(archive) as z:
            for member in z.infolist():
                if stat.S_ISLNK(member.external_attr >> 16):
                    raise ValueError("Archive symlinks are disallowed")
                path = target(member.filename, member.file_size)
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member, pwd=password.encode() if password else None) as source, path.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive) as tar:
            for member in tar:
                if not (member.isfile() or member.isdir()):
                    raise ValueError("Only regular files and directories are allowed in tar archives")
                path = target(member.name, member.size)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, path.open("wb") as sink:
                    shutil.copyfileobj(source, sink)
    else:
        raise ValueError("Unsupported archive; use ZIP, TAR, TAR.GZ, or pre-extracted local data")
    write_json(staging / ".bn5212_archive.json", {"sha256": expected_hash})
    os.replace(staging, destination)


def download_manifest(manifest_path):
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    root = Path(os.path.expandvars(manifest["destination_root"])).expanduser()
    root = (manifest_path.parent / root).resolve()
    for i, entry in enumerate(manifest["downloads"], 1):
        destination = safe_path(root, entry["destination"])
        print(f"Download {i}/{len(manifest['downloads'])}: checksum verification required")
        fetch_file(entry["url"], destination, entry["sha256"])
        if entry.get("extract_to"):
            extract_archive(destination, safe_path(root, entry["extract_to"]), entry["sha256"].lower(), int(entry["max_extract_bytes"]))
