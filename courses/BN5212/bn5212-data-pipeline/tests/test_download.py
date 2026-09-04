import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from src.data.download import extract_archive, fetch_file
from src.data.io import sha256


def test_safe_zip_and_reuse(tmp_path):
    archive = tmp_path / "safe.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("hosp/admissions.csv", "synthetic,only\n")
    dest = tmp_path / "unpacked"
    extract_archive(archive, dest, sha256(archive), 1000)
    assert (dest / "hosp/admissions.csv").read_text() == "synthetic,only\n"
    extract_archive(archive, dest, sha256(archive), 1000)


@pytest.mark.parametrize("filename", ["../escape", "..\\escape", "/absolute", "C:/escape"])
def test_reject_zip_traversal(tmp_path, filename):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(filename, "x")
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out", sha256(archive), 1000)
    assert not (tmp_path / "escape").exists()


def test_reject_tar_symlink(tmp_path):
    archive = tmp_path / "bad.tar"
    with tarfile.open(archive, "w") as t:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE; info.linkname = "../escape"
        t.addfile(info)
    with pytest.raises(ValueError):
        extract_archive(archive, tmp_path / "out", sha256(archive), 1000)


def test_budget_and_checksum(tmp_path):
    archive = tmp_path / "big.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("x", "12345")
    with pytest.raises(ValueError, match="budget"):
        extract_archive(archive, tmp_path / "out", sha256(archive), 4)
    fetch_file("https://example.org/never-contacted", archive, sha256(archive))
    with pytest.raises(ValueError, match="checksum"):
        fetch_file("https://example.org/never-contacted", archive, "0" * 64)


def test_resume_download_with_mock_http(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 206
        headers = {"Content-Range": "bytes 3-5/6"}
    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Range") == "bytes=3-"
            return Response(b"def")
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    dest = tmp_path / "data"
    dest.with_name("data.part").write_bytes(b"abc")
    fetch_file("https://example.org/synthetic", dest, hashlib.sha256(b"abcdef").hexdigest())
    assert dest.read_bytes() == b"abcdef"


def test_auth_is_bound_to_one_host(tmp_path, monkeypatch):
    class Response(io.BytesIO):
        status = 200
        headers = {}
    class Opener:
        def open(self, request, timeout):
            assert request.get_header("Authorization") is None
            return Response(b"synthetic")
    monkeypatch.setenv("DATA_HTTP_USERNAME", "synthetic-user")
    monkeypatch.setenv("DATA_HTTP_PASSWORD", "synthetic-test-secret")
    monkeypatch.setenv("DATA_HTTP_AUTH_HOST", "authorized.example")
    monkeypatch.setattr("urllib.request.build_opener", lambda *args: Opener())
    fetch_file("https://different.example/synthetic", tmp_path / "data", hashlib.sha256(b"synthetic").hexdigest())
