#!/usr/bin/env python3
"""Validate the PCMEA input layout and feature dimensions before training."""

import argparse
import pickle
from pathlib import Path

import numpy as np


EXPECTED = {
    "FB15K_DB15K": {
        "entities": 27793,
        "alignments": 12846,
        "triples": 691241,
        "images": 26281,
        "attributes": 23478,
    },
    "FB15K_YAGO15K": {
        "entities": 30355,
        "alignments": 11199,
        "triples": 715099,
        "images": 24638,
        "attributes": 27574,
    },
}


def nonempty_lines(path: Path) -> int:
    with path.open(encoding="utf-8") as stream:
        return sum(bool(line.strip()) for line in stream)


def load_feature(path: Path, expected_count: int, expected_dim: int) -> None:
    with path.open("rb") as stream:
        value = pickle.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"{path}: expected a dict, got {type(value).__name__}")
    if len(value) != expected_count:
        raise ValueError(f"{path}: expected {expected_count} entries, got {len(value)}")
    dimensions = {np.asarray(vector).reshape(-1).shape[0] for vector in value.values()}
    if dimensions != {expected_dim}:
        raise ValueError(f"{path}: expected dimension {expected_dim}, got {sorted(dimensions)}")
    if not all(np.isfinite(np.asarray(vector)).all() for vector in value.values()):
        raise ValueError(f"{path}: contains NaN or infinite values")
    print(f"OK  {path.name}: {len(value)} x {expected_dim}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=sorted(EXPECTED))
    parser.add_argument("--root", type=Path, default=Path("data/mmkb-datasets"))
    args = parser.parse_args()

    directory = args.root / args.dataset
    expected = EXPECTED[args.dataset]
    required_text = [
        "ent_ids_1", "ent_ids_2", "ill_ent_ids", "triples_1", "triples_2",
        "training_attrs_1", "training_attrs_2",
    ]
    missing = [name for name in required_text if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory}: missing {', '.join(missing)}")

    entities = nonempty_lines(directory / "ent_ids_1") + nonempty_lines(directory / "ent_ids_2")
    alignments = nonempty_lines(directory / "ill_ent_ids")
    triples = nonempty_lines(directory / "triples_1") + nonempty_lines(directory / "triples_2")
    observed = {"entities": entities, "alignments": alignments, "triples": triples}
    for key, value in observed.items():
        if value != expected[key]:
            raise ValueError(f"{directory}: expected {expected[key]} {key}, got {value}")
        print(f"OK  {key}: {value}")

    image_name = f"{args.dataset}_id_img_feature_dict.pkl"
    load_feature(directory / image_name, expected["images"], 4096)
    load_feature(directory / "attribute_feature_dict.pkl", expected["attributes"], 768)
    load_feature(directory / "triples_feature_dict.pkl", expected["entities"], 768)
    load_feature(directory / "attribute_roberta_base_feature_all.pkl", expected["attributes"], 768)
    load_feature(directory / "relation_t5_feature_all.pkl", expected["entities"], 768)
    print(f"PASS {args.dataset}")


if __name__ == "__main__":
    main()
