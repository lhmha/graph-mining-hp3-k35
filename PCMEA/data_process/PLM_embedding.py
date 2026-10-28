#!/usr/bin/env python3
"""Generate PCMEA relation/attribute embeddings with the paper's base PLMs."""
import argparse
import json
import os
import pickle
import re
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, T5EncoderModel


def text_process(values):
    text = " ".join(values).replace("<", "").replace(">", ";")
    text = text.replace("http://rdf.freebase.com/ns/", "").replace("http://dbpedia.org/", "")
    text = text.replace("_", " ").replace(".", " ")
    return re.sub("[A-Z]", lambda match: " " + match.group(0), text)


def read_attribute_content(paths):
    result = {}
    for path in paths:
        with open(path, encoding="utf-8") as fr:
            for line in fr:
                fields = line.rstrip("\n").split("\t")
                result[fields[0]] = fields[1:]
    return result


def entity_ids(paths):
    result = {}
    for path in paths:
        with open(path, encoding="utf-8") as fr:
            for line in fr:
                idx, entity = line.rstrip("\n").split("\t")[:2]
                result[entity] = int(idx)
    return result


def load_texts(args):
    if args.kind == "relation":
        source = json.load(open(args.input_json, encoding="utf-8"))
        result = {}
        for key, text in source.items():
            words = text.replace("/", "").split()
            result[int(key)] = " ".join(dict.fromkeys(words))[:2048]
        return result
    source = read_attribute_content(args.attribute_files)
    return {idx: text_process(source[entity]) for entity, idx in entity_ids(args.ent_ids).items()
            if entity in source}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("relation", "attribute"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-json")
    parser.add_argument("--attribute-files", nargs="+")
    parser.add_argument("--ent-ids", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model_class = T5EncoderModel if args.kind == "relation" else AutoModel
    model = model_class.from_pretrained(args.model, torch_dtype=torch.float16).to(device).eval()
    texts = load_texts(args)
    features = {}
    if args.resume and os.path.exists(args.output):
        features = pickle.load(open(args.output, "rb"))
    with torch.inference_mode():
        for count, (key, text) in enumerate(texts.items(), 1):
            if key in features:
                continue
            inputs = tokenizer([text], return_tensors="pt", truncation=True,
                               max_length=args.max_length).to(device)
            output = model(**inputs).last_hidden_state
            vector = output.mean(dim=1).squeeze(0).float().cpu().numpy()
            features[key] = np.resize(vector, (768,)).astype(np.float32)
            if count % 500 == 0:
                pickle.dump(features, open(args.output, "wb"), pickle.HIGHEST_PROTOCOL)
                print(f"embedded {count}/{len(texts)}", flush=True)
    pickle.dump(features, open(args.output, "wb"), pickle.HIGHEST_PROTOCOL)
    print(f"saved {len(features)} features of dimension 768 to {args.output}")


if __name__ == "__main__":
    main()
