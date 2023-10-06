from pathlib import Path


def data():
    return Path(__file__).resolve().parent / "data"
