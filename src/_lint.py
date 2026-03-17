# This file was developed with the assistance of Claude Code and Opus 4.6.

"""Wrapper so `uv run lint` lints the entire project with no extra args."""

import sys

from pylint import run_pylint

TARGETS = ["src/", "tests/", "main.py", "add_account.py"]


def main():
    sys.argv = sys.argv[:1] + TARGETS
    run_pylint()
