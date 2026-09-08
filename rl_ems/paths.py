"""Repository paths shared by executable workflows."""
from contextlib import contextmanager
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def repository_directory():
    """Support the research engine's relative data paths, restoring caller cwd."""
    previous = Path.cwd()
    os.chdir(ROOT)
    try:
        yield ROOT
    finally:
        os.chdir(previous)
