from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_publication_files_have_no_private_identity_or_host_paths() -> None:
    forbidden = tuple(
        bytes.fromhex(value).decode()
        for value in (
            "626a6f65726e",
            "626ac3b6726e",
            "656c6c656e736f686e",
            "756e696c656f62656e",
            "6d6f6e74616e756e6976657273",
            "6370735f",
            "6370732d",
            "6769746875622e636f6d2f626a6f65726e656c6c656e7331",
            "676863722e696f2f626a6f65726e656c6c656e7331",
            "2f686f6d652f",
            "2f6d6e742f",
            "6370732d776b73746e",
            "73706c61746f6772617068",
            "7065796d616e",
            "2f646174612f74756d5f7065796d616e2f",
        )
    )
    excluded = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".cache",
        "output",
    }
    offenders = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or not excluded.isdisjoint(path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if any(token in text for token in forbidden):
            offenders.append(path.relative_to(ROOT).as_posix())

    assert not offenders, offenders
