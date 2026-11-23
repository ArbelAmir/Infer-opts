#!/usr/bin/env python3
"""Depth pruning without fine-tuning: write a copy of a GGUF with some transformer blocks removed.

  PYTHONPATH=engines/llama.cpp/gguf-py harness/prune_layers.py in.gguf out.gguf --drop 12-15

Remaining blocks are renumbered and <arch>.block_count is updated; tensor data is copied bit-for-bit.
Qwen3.5 interleaves 3 linear-attention blocks with 1 full-attention block (full_attention_interval = 4)
and the engine derives the block type from the index, so drop whole groups of 4 aligned to that pattern.
"""
import argparse, re, sys

import numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, Keys


def parse_drop(spec):
    out = set()
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--drop", required=True, help="blocks to remove, e.g. 12-15 or 8-11,16-19")
    a = ap.parse_args()
    drop = parse_drop(a.drop)
    r = GGUFReader(a.src)
    arch = r.fields[Keys.General.ARCHITECTURE].contents()
    n_blocks = r.fields[f"{arch}.block_count"].contents()
    keep = [i for i in range(n_blocks) if i not in drop]
    remap = {old: new for new, old in enumerate(keep)}
    w = GGUFWriter(a.dst, arch=arch, endianess=r.endianess)
    for name, f in r.fields.items():
        if name == Keys.General.ARCHITECTURE or name.startswith("GGUF."):
            continue
        vt = f.types[0]
        sub = f.types[-1] if vt == GGUFValueType.ARRAY else None
        val = f.contents()
        if name == f"{arch}.block_count":
            val = len(keep)
        w.add_key_value(name, val, vt, sub_type=sub)
    pat = re.compile(r"^blk\.(\d+)\.(.*)$")
    n_drop = 0
    for t in r.tensors:
        name = t.name
        m = pat.match(name)
        if m:
            i = int(m.group(1))
            if i in drop:
                n_drop += 1
                continue
            name = f"blk.{remap[i]}.{m.group(2)}"
        w.add_tensor_info(name, [int(d) for d in reversed(t.shape)], np.dtype(t.data.dtype), int(t.n_bytes), t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in r.tensors:
        m = pat.match(t.name)
        if m and int(m.group(1)) in drop:
            continue
        w.write_tensor_data(t.data)
    w.close()
    print(f"{a.src}: {n_blocks} blocks -> {len(keep)} (dropped {sorted(drop)}, {n_drop} tensors)", file=sys.stderr)


if __name__ == "__main__":
    main()
