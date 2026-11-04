"""Download the harmful / harmless instruction sets with hash checks and build fixed splits.

Harmful:  AdvBench "harmful_behaviors.csv" (Zou et al., 2023), 520 instructions,
          fetched from the llm-attacks repository at a pinned commit.
Harmless: Stanford Alpaca "alpaca_data.json" (Taori et al., 2023), restricted to the
          instructions whose `input` field is empty (as in Arditi et al., 2024),
          fetched from the stanford_alpaca repository at a pinned commit.

Arditi et al. also use MaliciousInstruct, TDC2023 and HarmBench prompts; this
replication uses AdvBench only (public, single file, stable). The exact sampled
splits are written to data/splits_seed{seed}.json so that a run is reproducible
even if the upstream files move.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen


@dataclass(frozen=True)
class Source:
    url: str
    sha256: str
    filename: str


SOURCES: dict[str, Source] = {
    "harmful": Source(
        url=(
            "https://raw.githubusercontent.com/llm-attacks/llm-attacks/"
            "098262edf85f807224e70ecd87b9d83716bf6b73/data/advbench/harmful_behaviors.csv"
        ),
        sha256="6cd1a5c63c07610d7eb67307772ee5606017ee950b5770ab288a2c487489d3e1",
        filename="advbench_harmful_behaviors.csv",
    ),
    "harmless": Source(
        url=(
            "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
            "761dc5bfbdeeffa89b8bff5d038781a4055f796a/alpaca_data.json"
        ),
        sha256="2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419",
        filename="alpaca_data.json",
    ),
    # Transfer sets for the extensions (not used to fit the direction).
    "harmbench": Source(
        url=(
            "https://raw.githubusercontent.com/centerforaisafety/HarmBench/"
            "8e1604d1171fe8a48d8febecd22f600e462bdcdd/data/behavior_datasets/"
            "harmbench_behaviors_text_all.csv"
        ),
        sha256="8d81accedd38eaaf8b760618622bb888417d1fd0c86eba65c427a16f1cbb4afc",
        filename="harmbench_behaviors_text_all.csv",
    ),
    "jbb_harmful": Source(
        url=(
            "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/"
            "886acc352a31533ffbcf4ef22c744658688086fc/data/harmful-behaviors.csv"
        ),
        sha256="4a8ec6832056b631eb092dccc60d37a61c3d441268268888b3d006288afeffa1",
        filename="jbb_harmful_behaviors.csv",
    ),
    "jbb_benign": Source(
        url=(
            "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors/resolve/"
            "886acc352a31533ffbcf4ef22c744658688086fc/data/benign-behaviors.csv"
        ),
        sha256="3cda234d21a991fa309bbfea4b6d9dae31ccdf8e9d452424b6a983e4fdc33468",
        filename="jbb_benign_behaviors.csv",
    ),
}

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    """Download one source file (if missing) and verify its SHA-256. Raises on mismatch."""
    src = SOURCES[name]
    raw_dir.mkdir(parents=True, exist_ok=True)
    path = raw_dir / src.filename
    if not path.exists():
        with urlopen(src.url, timeout=60) as r, open(path, "wb") as f:
            f.write(r.read())
    digest = sha256_of(path)
    if digest != src.sha256:
        path.unlink(missing_ok=True)
        raise RuntimeError(
            f"SHA-256 mismatch for {src.filename}: expected {src.sha256}, got {digest}. "
            "The file was deleted; re-run to download again."
        )
    return path


def load_harmful(raw_dir: Path = DEFAULT_RAW_DIR) -> list[str]:
    path = download("harmful", raw_dir)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [row["goal"].strip() for row in rows]


def load_harmless(raw_dir: Path = DEFAULT_RAW_DIR) -> list[str]:
    path = download("harmless", raw_dir)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    seen: set[str] = set()
    out: list[str] = []
    for row in rows:
        if row.get("input", "").strip():
            continue
        text = row["instruction"].strip()
        if text and text not in seen:
            seen.add(text)
            out.append(text)
    return out


def load_harmless_pairs(raw_dir: Path = DEFAULT_RAW_DIR) -> list[tuple[str, str]]:
    """(instruction, reference output) for the empty-input Alpaca rows, deduplicated by instruction."""
    path = download("harmless", raw_dir)
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []
    for row in rows:
        if row.get("input", "").strip():
            continue
        text = row["instruction"].strip()
        if text and text not in seen:
            seen.add(text)
            out.append((text, row["output"].strip()))
    return out


def load_harmbench(raw_dir: Path = DEFAULT_RAW_DIR) -> list[str]:
    """HarmBench text behaviors of the 'standard' functional category (no context string)."""
    path = download("harmbench", raw_dir)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [
        r["Behavior"].strip()
        for r in rows
        if r["FunctionalCategory"] == "standard" and not r["ContextString"].strip()
    ]


def load_jbb(which: str, raw_dir: Path = DEFAULT_RAW_DIR, exclude_advbench: bool = True) -> list[str]:
    """JailbreakBench behaviors ('harmful' or 'benign'). The harmful set's rows sourced from
    AdvBench are dropped by default because AdvBench is what the direction is fitted on."""
    path = download(f"jbb_{which}", raw_dir)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if exclude_advbench:
        rows = [r for r in rows if "advbench" not in r.get("Source", "").lower()]
    return [r["Goal"].strip() for r in rows]


def make_transfer_sets(n: int = 64, seed: int = 0, raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    """Fixed-seed samples of the transfer sets used by the extensions."""
    rng = random.Random(seed)
    pools = {
        "harmbench_standard": load_harmbench(raw_dir),
        "jbb_harmful": load_jbb("harmful", raw_dir),
        "jbb_benign": load_jbb("benign", raw_dir),
    }
    out = {"seed": seed, "n": n, "sources": {k: {"url": SOURCES[k].url, "sha256": SOURCES[k].sha256}
                                              for k in ("harmbench", "jbb_harmful", "jbb_benign")}}
    for name, pool in pools.items():
        out[name] = [pool[i] for i in sorted(rng.sample(range(len(pool)), min(n, len(pool))))]
    return out


def get_transfer_sets(path: Path, n: int = 64, seed: int = 0, raw_dir: Path = DEFAULT_RAW_DIR) -> dict:
    if path.exists():
        sets = load_splits(path)
        if (sets["seed"], sets["n"]) != (seed, n):
            raise ValueError(f"{path} was built with a different seed/size than requested")
        return sets
    sets = make_transfer_sets(n, seed, raw_dir)
    save_splits(sets, path)
    return sets


def make_splits(
    harmful: list[str],
    harmless: list[str],
    n_train: int = 128,
    n_val: int = 32,
    seed: int = 0,
) -> dict:
    """Sample disjoint train / val sets per class with a fixed seed (paper: 128 train, 32 val)."""
    rng = random.Random(seed)
    harmful_idx = rng.sample(range(len(harmful)), n_train + n_val)
    harmless_idx = rng.sample(range(len(harmless)), n_train + n_val)
    return {
        "seed": seed,
        "n_train": n_train,
        "n_val": n_val,
        "sources": {k: {"url": v.url, "sha256": v.sha256} for k, v in SOURCES.items()},
        "harmful_train": [harmful[i] for i in harmful_idx[:n_train]],
        "harmful_val": [harmful[i] for i in harmful_idx[n_train:]],
        "harmless_train": [harmless[i] for i in harmless_idx[:n_train]],
        "harmless_val": [harmless[i] for i in harmless_idx[n_train:]],
    }


def save_splits(splits: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=1, ensure_ascii=False)


def load_splits(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_splits(
    path: Path,
    n_train: int = 128,
    n_val: int = 32,
    seed: int = 0,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> dict:
    """Load the saved splits if present, otherwise download the sources, sample and save."""
    if path.exists():
        splits = load_splits(path)
        if (splits["seed"], splits["n_train"], splits["n_val"]) != (seed, n_train, n_val):
            raise ValueError(f"{path} was built with different seed/sizes than requested")
        return splits
    splits = make_splits(load_harmful(raw_dir), load_harmless(raw_dir), n_train, n_val, seed)
    save_splits(splits, path)
    return splits


def main() -> None:
    p = argparse.ArgumentParser(description="Download datasets (with hash check) and write splits.")
    p.add_argument("--out", type=Path, default=Path("data/splits_seed0.json"))
    p.add_argument("--n-train", type=int, default=128)
    p.add_argument("--n-val", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    splits = make_splits(load_harmful(), load_harmless(), a.n_train, a.n_val, a.seed)
    save_splits(splits, a.out)
    print(f"wrote {a.out}: {a.n_train} train + {a.n_val} val per class, seed {a.seed}")


if __name__ == "__main__":
    main()
