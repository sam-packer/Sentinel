"""Tests for the collection status tracking."""

import json
import os

from src.collector import (
    STATUS_FILE,
    PID_FILE,
    CollectionStatus,
    _update_status,
    read_status,
    is_running,
)


class TestCollectionStatus:
    def test_update_and_read_status(self, tmp_path, monkeypatch):
        status_file = tmp_path / "status.json"
        monkeypatch.setattr("src.collector.STATUS_FILE", status_file)

        status = CollectionStatus(
            pid=12345,
            state="running",
            scraped=50,
            labeled=30,
            failed=2,
        )
        _update_status(status)

        assert status_file.exists()
        data = json.loads(status_file.read_text())
        assert data["pid"] == 12345
        assert data["state"] == "running"
        assert data["scraped"] == 50
        assert data["last_update"] is not None

    def test_read_status_returns_none_when_missing(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.collector.STATUS_FILE", tmp_path / "nonexistent.json")
        assert read_status() is None

    def test_read_status_roundtrip(self, tmp_path, monkeypatch):
        status_file = tmp_path / "status.json"
        monkeypatch.setattr("src.collector.STATUS_FILE", status_file)

        original = CollectionStatus(
            pid=99, state="enriching", scraped=100, enriched=40,
            labeled=38, failed=2, current_ticker="RTX",
        )
        _update_status(original)
        restored = read_status()

        assert restored.pid == 99
        assert restored.state == "enriching"
        assert restored.scraped == 100
        assert restored.enriched == 40
        assert restored.current_ticker == "RTX"

    def test_read_status_handles_corrupt_json(self, tmp_path, monkeypatch):
        status_file = tmp_path / "status.json"
        status_file.write_text("not valid json{{{")
        monkeypatch.setattr("src.collector.STATUS_FILE", status_file)
        assert read_status() is None

    def test_is_running_false_when_no_pid_file(self, tmp_path, monkeypatch):
        monkeypatch.setattr("src.collector.PID_FILE", tmp_path / "no.pid")
        assert is_running() is False

    def test_is_running_false_when_pid_dead(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "collect.pid"
        pid_file.write_text("999999999")  # unlikely to be a real PID
        monkeypatch.setattr("src.collector.PID_FILE", pid_file)
        assert is_running() is False
        assert not pid_file.exists()  # should clean up stale PID file

    def test_is_running_true_for_own_pid(self, tmp_path, monkeypatch):
        pid_file = tmp_path / "collect.pid"
        pid_file.write_text(str(os.getpid()))
        monkeypatch.setattr("src.collector.PID_FILE", pid_file)
        assert is_running() is True
