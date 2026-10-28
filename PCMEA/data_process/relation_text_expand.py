#!/usr/bin/env python3
"""Expand MMKB relation triples into per-entity text (PCMEA preprocessing)."""
import argparse
import json


def entity_id_read(file_paths):
    ent2id_dict = {}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as fr:
            for line in fr:
                idx, entity = line.rstrip("\n").split("\t")[:2]
                ent2id_dict[entity] = int(idx)
    return ent2id_dict


def relation_id_read(entity_id_dict, entity_triple_paths):
    rel2id_dict = {}
    for entity_triple_path in entity_triple_paths:
        with open(entity_triple_path, "r", encoding="utf-8") as fr:
            for line in fr:
                params = line.rstrip("\n").split(" ")
                if len(params) < 3:
                    continue
                e1, relation, e2 = params[:3]
                for entity in (e1, e2):
                    if entity in entity_id_dict:
                        idx = entity_id_dict[entity]
                        rel2id_dict[idx] = (rel2id_dict.get(idx, "") + " " + relation).strip()
    return rel2id_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ent-ids", nargs="+", required=True)
    parser.add_argument("--entity-triples", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ent2id_dict = entity_id_read(args.ent_ids)
    relation_dict = relation_id_read(ent2id_dict, args.entity_triples)
    with open(args.output, "w", encoding="utf-8") as fw:
        json.dump(relation_dict, fw)
    print(f"mapped {len(relation_dict)} of {len(ent2id_dict)} entities")


if __name__ == "__main__":
    main()
