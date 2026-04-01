import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]

_CONFIG_PATH = Path(__file__).parent / "config.toml"


def load() -> dict:
    if not _CONFIG_PATH.exists():
        print(
            f"[error] config.toml not found at {_CONFIG_PATH}\n"
            "Copy config.toml.example to config.toml and fill in your settings.",
            file=sys.stderr,
        )
        sys.exit(1)
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)
