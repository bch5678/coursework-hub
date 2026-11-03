"""Download the authorized course ZIPs with password authentication and resume."""
from __future__ import annotations

import argparse
import getpass
import hashlib
from http.cookiejar import LWPCookieJar
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import time
import urllib.parse
import zipfile
import zlib

PROJECT_DIR = Path(__file__).resolve().parents[1]
TABLES = (
    "hosp/admissions.csv", "hosp/patients.csv", "hosp/d_labitems.csv",
    "icu/d_items.csv", "icu/icustays.csv", "icu/chartevents.csv",
    "hosp/labevents.csv", "hosp/omr.csv",
)


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2)
        stream.write("\n")
    temporary.replace(path)
    path.chmod(0o600)


def dropbox_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.dropbox.com", "dropbox.com"} or parsed.username or parsed.password:
        raise ValueError("The private source config must contain HTTPS Dropbox sharing URLs")
    return url


def authorize(session, source: dict, password: str) -> None:
    page = session.get(dropbox_url(source["shared_url"]), timeout=(30, 90))
    page.raise_for_status()
    if urllib.parse.urlsplit(page.url).path != "/sm/password":
        if "Password Required" in page.text:
            raise ValueError("Dropbox requires a sharing password; refresh the private source config")
        return
    if not password:
        raise ValueError("A Dropbox sharing password is required")
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(page.url).query)
    content_id = query.get("content_id", [""])[0]
    csrf = next((cookie.value for cookie in session.cookies if cookie.name == "t"), "")
    if not content_id or not csrf:
        raise ValueError("Dropbox password form changed; no data was downloaded")
    response = session.post(
        "https://www.dropbox.com/sm/auth",
        data={"content_id": content_id, "password": password,
              "url": query.get("cont", [source["shared_url"]])[0], "t": csrf},
        headers={"Origin": "https://www.dropbox.com", "Referer": page.url,
                 "X-Requested-With": "XMLHttpRequest"}, timeout=(30, 90),
    )
    response.raise_for_status()
    if response.json().get("status") != "authed":
        raise ValueError("Dropbox did not accept the sharing password")
    print("Dropbox shared folder unlocked.", flush=True)


def download_file(session, item: dict, root: Path, *, retries: int = 20,
                  retry_delay: float = 10, restart: bool = False) -> Path:
    import requests

    relative = PurePosixPath(item["destination"])
    if relative.is_absolute() or ".." in relative.parts or "\\" in str(relative):
        raise ValueError("Unsafe download destination")
    destination = root / relative
    size = int(item["size"])
    if size <= 0:
        raise ValueError("Source config must specify a positive remote file size")
    url = dropbox_url(item["url"])
    identity = hashlib.sha256(url.encode()).hexdigest()
    metadata_path = destination.with_name(destination.name + ".download.local.json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if destination.exists():
        print(f"Checking completed ZIP: {destination.name}", flush=True)
        if destination.stat().st_size == size and metadata.get("source_url_sha256") == identity and metadata.get("sha256"):
            digest = checksum(destination)
            if digest != metadata["sha256"]:
                raise ValueError(f"Completed ZIP failed its recorded local SHA-256: {destination}")
            if item.get("sha256") and digest.lower() != item["sha256"].lower():
                raise ValueError(f"Publisher SHA-256 mismatch for {destination.name}")
            metadata["complete"] = True
            metadata["publisher_sha256_verified"] = bool(item.get("sha256"))
            write_private_json(metadata_path, metadata)
            print(f"SKIP verified completed file: {destination.name}", flush=True)
            return destination
        raise ValueError(f"Existing final file lacks matching verification metadata: {destination}")
    partial = destination.with_name(destination.name + ".part")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if partial.exists() and not restart and metadata.get("source_url_sha256") != identity:
        raise ValueError(f"Partial file belongs to an unknown source: {partial}; inspect it before --restart")
    if restart and partial.exists():
        with partial.open("wb"):
            pass
        metadata = {}
    metadata.update(source_url_sha256=identity, expected_size=size)
    write_private_json(metadata_path, metadata)
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Partial file exceeds expected size: {partial}")
        if offset == size:
            break
        if shutil.disk_usage(destination.parent).free < size - offset:
            raise OSError(f"Not enough disk space for {destination.name}")
        headers = {"Range": f"bytes={offset}-", "Accept-Encoding": "identity"}
        validator = metadata.get("etag") or metadata.get("last_modified")
        if offset and validator:
            headers["If-Range"] = validator
        started = time.monotonic()
        reported = started
        print(f"Downloading {destination.name}: {offset / 2**30:.2f}/{size / 2**30:.2f} GiB", flush=True)
        try:
            with session.get(url, headers=headers, stream=True, timeout=(30, 120)) as response:
                if response.status_code in {408, 429, 500, 502, 503, 504}:
                    raise requests.ConnectionError(f"Temporary HTTP {response.status_code}")
                if response.status_code not in {200, 206}:
                    raise ValueError(f"Download HTTP {response.status_code}; rerun to refresh authentication")
                if "text/html" in response.headers.get("Content-Type", "").lower():
                    raise ValueError("Dropbox returned an HTML page instead of ZIP data; check sharing access")
                if response.status_code == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if not match or int(match[1]) != offset or int(match[3]) != size:
                        raise ValueError("Remote range/size changed; partial ZIP retained")
                elif offset:
                    raise ValueError("Server ignored resume range; partial retained. Use --restart to start it over")
                for key, header in (("etag", "ETag"), ("last_modified", "Last-Modified")):
                    value = response.headers.get(header)
                    if value:
                        if offset and metadata.get(key) and metadata[key] != value:
                            raise ValueError("Remote ZIP changed since the partial download was created")
                        metadata[key] = value
                write_private_json(metadata_path, metadata)
                received = offset
                with partial.open("ab" if offset else "wb") as stream:
                    for block in response.iter_content(1024 * 1024):
                        if not block:
                            continue
                        if received + len(block) > size:
                            raise ValueError("Server sent more bytes than the advertised ZIP size")
                        stream.write(block)
                        received += len(block)
                        now = time.monotonic()
                        if now - reported >= 10:
                            rate = (received - offset) / max(now - started, 0.001)
                            eta = (size - received) / max(rate, 1.0) / 60
                            print(f"{destination.name}: {100 * received / size:.1f}% | "
                                  f"{rate / 2**20:.1f} MiB/s | ETA {eta:.1f} min", flush=True)
                            reported = now
                if received != size:
                    raise requests.ConnectionError("Download response ended before the expected file size")
            break
        except requests.RequestException as error:
            if attempt >= retries:
                raise RuntimeError(f"Download retries exhausted ({type(error).__name__}); rerun to resume {partial}") from None
            delay = min(60, retry_delay * (attempt + 1))
            print(f"Network interruption ({type(error).__name__}); retry {attempt + 1}/{retries} in {delay:g}s", flush=True)
            time.sleep(delay)
    print(f"Checking ZIP directory and computing SHA-256: {destination.name}", flush=True)
    with zipfile.ZipFile(partial) as archive:
        archive.infolist()
    digest = checksum(partial)
    if item.get("sha256") and digest.lower() != item["sha256"].lower():
        raise ValueError(f"Publisher SHA-256 mismatch for {destination.name}; partial retained")
    metadata.update(sha256=digest, publisher_sha256_verified=bool(item.get("sha256")), complete=False)
    write_private_json(metadata_path, metadata)
    partial.replace(destination)
    metadata["complete"] = True
    write_private_json(metadata_path, metadata)
    print(f"DONE {destination.name} | SHA-256 {digest}", flush=True)
    return destination


def extract_inputs(archive_path: Path, output: Path, *, kind: str) -> None:
    journal_path = output / ".extraction.local.json"
    journal = json.loads(journal_path.read_text()) if journal_path.is_file() else {"entries": {}}
    expected_source = checksum(archive_path)
    if journal.get("archive_sha256") not in (None, expected_source):
        raise ValueError(f"Extraction directory belongs to a different archive: {output}")
    journal["archive_sha256"] = expected_source
    selected = []
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            name = entry.filename.replace("\\", "/")
            if entry.is_dir():
                continue
            target = None
            if kind == "cxr":
                match = re.search(r"(?:^|/)(?:files/)?(p\d{2}/p\d+/s\d+/[^/]+\.dcm)$", name, re.I)
                if match:
                    target = "files/" + match[1]
            else:
                target = next((table for table in TABLES if name == table or name.endswith("/" + table)), None)
            if target:
                if stat.S_ISLNK(entry.external_attr >> 16):
                    raise ValueError("Archive symlinks are not supported")
                selected.append((entry, target))
        targets = [target for _, target in selected]
        if not selected or len(set(targets)) != len(targets):
            raise ValueError("No matching inputs, or duplicate flattened paths in the ZIP")
        if kind == "iv":
            missing = set(TABLES) - set(targets)
            if missing:
                raise ValueError(f"MIMIC-IV archive is missing required tables: {sorted(missing)}")
        output.mkdir(parents=True, exist_ok=True)
        remaining = sum(entry.file_size for entry, target in selected
                        if not (output / target).is_file())
        if shutil.disk_usage(output).free < remaining:
            raise OSError(f"Not enough disk space to extract {archive_path.name}")
        password = os.environ.get("DATA_ARCHIVE_PASSWORD")
        for position, (entry, target) in enumerate(selected, 1):
            destination = output / target
            known = journal["entries"].get(target)
            if destination.is_file() and known == [entry.file_size, entry.CRC, destination.stat().st_mtime_ns]:
                continue
            if destination.exists():
                if destination.stat().st_size != entry.file_size:
                    raise ValueError(f"Existing extracted file has a different size: {destination}")
                crc = 0
                with destination.open("rb") as stream:
                    for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                        crc = zlib.crc32(block, crc)
                if crc != entry.CRC:
                    raise ValueError(f"Existing extracted file failed CRC: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                partial = destination.with_name(destination.name + ".part")
                print(f"Extract {position}/{len(selected)}: {target}", flush=True)
                with archive.open(entry, pwd=password.encode() if password else None) as source, partial.open("wb") as sink:
                    shutil.copyfileobj(source, sink, 4 * 1024 * 1024)
                partial.replace(destination)
            journal["entries"][target] = [entry.file_size, entry.CRC, destination.stat().st_mtime_ns]
            write_private_json(journal_path, journal)
    print(f"Extracted/verified {len(selected)} inputs: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-config", default=str(PROJECT_DIR / "data/dropbox_source.local.json"))
    parser.add_argument("--output-dir", default=str(PROJECT_DIR / "data/downloads"))
    parser.add_argument("--only", choices=("all", "cxr", "iv"), default="all")
    parser.add_argument("--list-only", action="store_true", help="Show cached file names/sizes without any network access")
    parser.add_argument("--extract", action="store_true", help="Extract DICOMs and needed MIMIC-IV tables after downloads")
    parser.add_argument("--data-root", default=str(PROJECT_DIR / "data"))
    parser.add_argument("--retries", type=int, default=20)
    parser.add_argument("--retry-delay", type=float, default=10)
    parser.add_argument("--restart", action="store_true", help="Discard only the selected incomplete .part files and start over")
    args = parser.parse_args()
    if args.retries < 0 or args.retry_delay < 0:
        parser.error("retry count and delay must be nonnegative")
    source_path = Path(args.source_config).resolve()
    source = json.loads(source_path.read_text())
    selected = [item for item in source["downloads"] if args.only == "all" or
                (item["destination"].startswith("MIMIC-CXR/") if args.only == "cxr" else
                 item["destination"].startswith("MIMIC-IV/"))]
    if not selected:
        raise ValueError("No matching files in the private source config")
    for item in selected:
        print(f"{item['destination']}: {int(item['size']) / 2**30:.2f} GiB", flush=True)
    print(f"Total ZIP size: {sum(int(item['size']) for item in selected) / 2**30:.2f} GiB", flush=True)
    if args.list_only:
        return
    try:
        import requests
    except ImportError:
        raise RuntimeError("Install the download extra: python -m pip install -e '.[download]'") from None
    password = os.environ.get("BN5212_DROPBOX_PASSWORD") or source.get("password", "")
    if not password and sys.stdin.isatty():
        password = getpass.getpass("Dropbox sharing password: ")
    session = requests.Session()
    cookies = LWPCookieJar(str(source_path.with_name("dropbox_cookies.local.txt")))
    if Path(cookies.filename).is_file():
        cookies.load(ignore_discard=True, ignore_expires=True)
    session.cookies = cookies
    authorize(session, source, password)
    cookies.save(ignore_discard=True, ignore_expires=True)
    Path(cookies.filename).chmod(0o600)
    output = Path(args.output_dir).resolve()
    for item in selected:
        archive_path = download_file(session, item, output, retries=args.retries,
                                     retry_delay=args.retry_delay, restart=args.restart)
        if args.extract:
            kind = "cxr" if item["destination"].startswith("MIMIC-CXR/") else "iv"
            target = Path(args.data_root) / ("mimic-cxr/2.1.0" if kind == "cxr" else "mimiciv")
            extract_inputs(archive_path, target, kind=kind)
    print(f"All requested downloads completed: {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume .part downloads.", file=sys.stderr)
        raise SystemExit(130)
    except Exception as error:
        if error.__class__.__module__.startswith("requests"):
            print(f"Download failed ({type(error).__name__}); check connection/access and rerun.", file=sys.stderr)
        else:
            print(f"Download failed: {error}", file=sys.stderr)
        raise SystemExit(1)
