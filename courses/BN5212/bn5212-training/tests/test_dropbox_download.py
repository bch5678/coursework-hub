"""Resumable file downloads, ZIP integrity and selective input extraction."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest
import requests

from bn5212_training.dropbox_download import (
    TABLES, checksum, download_file, extract_inputs, write_private_json,
)


def _zip_blob():
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        archive.writestr("fixture.txt", "synthetic download test")
    return stream.getvalue()


class Response:
    status_code = 206

    def __init__(self, body, offset, *, interrupt=False, start=None):
        self.body = body
        self.offset = offset
        self.interrupt = interrupt
        self.headers = {
            "Content-Type": "application/binary", "ETag": '"same-version"',
            "Content-Range": f"bytes {offset if start is None else start}-{len(body)-1}/{len(body)}",
        }

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def iter_content(self, chunk_size):
        remaining = self.body[self.offset:]
        if self.interrupt:
            yield remaining[:10]
            raise requests.ConnectionError("synthetic interruption")
        yield remaining


class Session:
    def __init__(self, body, *, interrupt=False, wrong_start=False, ignore_range=False):
        self.body = body
        self.headers = []
        self.interrupt = interrupt
        self.wrong_start = wrong_start
        self.ignore_range = ignore_range

    def get(self, url, *, headers, stream, timeout):
        self.headers.append(headers)
        offset = int(headers["Range"].split("=")[1].split("-")[0])
        response = Response(self.body, offset, interrupt=self.interrupt and len(self.headers) == 1,
                            start=0 if self.wrong_start else None)
        if self.ignore_range:
            response.status_code = 200
        return response


def _item(body):
    return {"url": "https://www.dropbox.com/scl/fi/test/file.zip?dl=1",
            "destination": "MIMIC-IV/fixture.zip", "size": len(body)}


def test_network_interruption_resumes_and_completed_file_is_verified(tmp_path):
    body = _zip_blob()
    session = Session(body, interrupt=True)
    path = download_file(session, _item(body), tmp_path, retries=1, retry_delay=0)
    assert path.read_bytes() == body
    assert session.headers[1]["Range"] == "bytes=10-"
    assert session.headers[1]["If-Range"] == '"same-version"'
    metadata = json.loads(path.with_name(path.name + ".download.local.json").read_text())
    assert metadata["sha256"] == hashlib.sha256(body).hexdigest()
    assert metadata["complete"]
    assert metadata["publisher_sha256_verified"] is False
    assert download_file(session, _item(body), tmp_path) == path
    assert len(session.headers) == 2


@pytest.mark.parametrize("case", ["changed_range", "ignored_range", "wrong_sha"])
def test_changed_or_invalid_download_does_not_publish_a_final_zip(tmp_path, case):
    body = _zip_blob()
    item = _item(body)
    destination = tmp_path / item["destination"]
    destination.parent.mkdir()
    if case != "wrong_sha":
        destination.with_name(destination.name + ".part").write_bytes(body[:10])
        write_private_json(destination.with_name(destination.name + ".download.local.json"), {
            "source_url_sha256": hashlib.sha256(item["url"].encode()).hexdigest(),
        })
    else:
        item["sha256"] = "0" * 64
    session = Session(body, wrong_start=case == "changed_range", ignore_range=case == "ignored_range")
    with pytest.raises(ValueError):
        download_file(session, item, tmp_path, retries=0)
    assert not destination.exists()
    assert destination.with_name(destination.name + ".part").exists()


def test_invalid_destination_is_rejected_before_network_access(tmp_path):
    body = _zip_blob()
    item = {**_item(body), "destination": "../outside.zip"}
    session = Session(body)
    with pytest.raises(ValueError, match="Unsafe"):
        download_file(session, item, tmp_path)
    assert not session.headers


def test_new_publisher_checksum_is_checked_for_an_existing_complete_file(tmp_path):
    body = _zip_blob()
    item = _item(body)
    path = download_file(Session(body), item, tmp_path)
    with pytest.raises(ValueError, match="Publisher SHA-256 mismatch"):
        download_file(Session(body), {**item, "sha256": "0" * 64}, tmp_path)
    assert path.read_bytes() == body


def test_selective_cxr_extraction_flattens_prefix_and_reuses_completed_files(tmp_path):
    source = tmp_path / "images.zip"
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("pack/files/p90/p90000000/s70000000/synthetic.dcm", b"synthetic DICOM bytes")
        archive.writestr("pack/reports/ignored.txt", "not required")
    output = tmp_path / "cxr"
    extract_inputs(source, output, kind="cxr")
    extracted = output / "files/p90/p90000000/s70000000/synthetic.dcm"
    modified = extracted.stat().st_mtime_ns
    assert extracted.is_file()
    assert not (output / "reports").exists()
    extract_inputs(source, output, kind="cxr")
    assert extracted.stat().st_mtime_ns == modified
    assert json.loads((output / ".extraction.local.json").read_text())["archive_sha256"] == checksum(source)


def test_mimic_extraction_keeps_required_tables_only(tmp_path):
    source = tmp_path / "tables.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for table in TABLES:
            archive.writestr("pack/mimiciv/3.1/" + table, "header\nsynthetic\n")
        archive.writestr("pack/mimiciv/3.1/hosp/unused.csv", "not needed")
    output = tmp_path / "mimic"
    extract_inputs(source, output, kind="iv")
    assert all((output / table).is_file() for table in TABLES)
    assert not (output / "hosp/unused.csv").exists()
