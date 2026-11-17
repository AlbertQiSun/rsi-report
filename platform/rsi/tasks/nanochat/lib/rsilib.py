# Copyright 2026 Recursive
# Copyright 2025 Andrej Karpathy
# SPDX-License-Identifier: Apache-2.0
# Modified for RSI-mini (2026-11-17): data/tokenizer/export paths come from the
# sandbox environment; the validation split and evaluate_bpb are removed (the
# sealed evaluator scores the exported model); export_model() added.
"""Read-only runtime library for the NanoChat task (mounted at /opt/rsi/lib).

What train.py gets from here:
  MAX_SEQ_LEN, TIME_BUDGET          constants (2048, 300 s of training by default)
  Tokenizer                          the fixed BPE tokenizer (do not change it)
  get_token_bytes(device)            bytes per token id (0 for special tokens)
  make_dataloader(tok, B, T, "train")  BOS-aligned best-fit packed batches from /data/train
  export_model(model, config)        REQUIRED before exit: writes the weights + config the evaluator scores
  time_left()                        seconds until the evaluator's hard wall-clock kill
"""

import json
import os
import pickle
import time

import pyarrow.parquet as pq
import torch

MAX_SEQ_LEN = int(os.environ.get("RSI_SEQ_LEN", "2048"))
TIME_BUDGET = float(os.environ.get("RSI_TIME_BUDGET", "300"))
WALL_LIMIT_S = float(os.environ.get("RSI_WALL_LIMIT_S", "0") or 0)
DEADLINE_UNIX = float(os.environ.get("RSI_DEADLINE_UNIX", "0") or 0)

DATA_DIR = os.environ.get("RSI_DATA_DIR", "/data/train")
TOKENIZER_DIR = os.environ.get("RSI_TOKENIZER_DIR", "/data/tokenizer")
EXPORT_DIR = os.environ.get("RSI_EXPORT_DIR", "/work/export")
BOS_TOKEN = "<|reserved_0|>"


def time_left():
    """Seconds until the evaluator kills this process (inf if unknown)."""
    return DEADLINE_UNIX - time.time() if DEADLINE_UNIX > 0 else float("inf")


def list_parquet_files():
    files = sorted(f for f in os.listdir(DATA_DIR) if f.endswith(".parquet") and not f.endswith(".tmp"))
    return [os.path.join(DATA_DIR, f) for f in files]


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    @classmethod
    def from_directory(cls, tokenizer_dir=None):
        with open(os.path.join(tokenizer_dir or TOKENIZER_DIR, "tokenizer.pkl"), "rb") as f:
            enc = pickle.load(f)
        return cls(enc)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text, prepend=None, num_threads=8):
        if prepend is not None:
            prepend_id = prepend if isinstance(prepend, int) else self.enc.encode_single_token(prepend)
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
        elif isinstance(text, list):
            ids = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in ids:
                    row.insert(0, prepend_id)
        else:
            raise ValueError(f"Invalid input type: {type(text)}")
        return ids

    def decode(self, ids):
        return self.enc.decode(ids)


def get_token_bytes(device="cpu"):
    path = os.path.join(TOKENIZER_DIR, "token_bytes.pt")
    with open(path, "rb") as f:
        return torch.load(f, map_location=device)


def _document_batches(split, tokenizer_batch_size=128):
    if split != "train":
        raise ValueError("Only the 'train' split exists inside the training sandbox; "
                         "the evaluator scores your exported model on held-out data.")
    parquet_paths = list_parquet_files()
    assert len(parquet_paths) > 0, f"No training shards found in {DATA_DIR}."
    epoch = 1
    while True:
        for filepath in parquet_paths:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                rg = pf.read_row_group(rg_idx)
                batch = rg.column('text').to_pylist()
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size], epoch
        epoch += 1


def make_dataloader(tokenizer, B, T, split="train", buffer_size=1000, device="cuda"):
    """BOS-aligned dataloader with best-fit packing. Every row starts with BOS;
    documents are packed best-fit, cropping the shortest doc when nothing fits.
    Yields (inputs, targets, epoch) with inputs/targets of shape (B, T) on ``device``."""
    assert split == "train", "only the train split is available"
    row_capacity = T + 1
    batches = _document_batches(split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer = []
    epoch = 1

    def refill_buffer():
        nonlocal epoch
        doc_batch, epoch = next(batches)
        token_lists = tokenizer.encode(doc_batch, prepend=bos_token)
        doc_buffer.extend(token_lists)

    use_cuda = str(device).startswith("cuda")
    row_buffer = torch.empty((B, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(2 * B * T, dtype=torch.long, pin_memory=use_cuda)
    gpu_buffer = torch.empty(2 * B * T, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[:B * T].view(B, T)
    cpu_targets = cpu_buffer[B * T:].view(B, T)
    inputs = gpu_buffer[:B * T].view(B, T)
    targets = gpu_buffer[B * T:].view(B, T)

    while True:
        for row_idx in range(B):
            pos = 0
            while pos < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - pos

                best_idx = -1
                best_len = 0
                for i, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = i
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, pos:pos + len(doc)] = torch.tensor(doc, dtype=torch.long)
                    pos += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda i: len(doc_buffer[i]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, pos:pos + remaining] = torch.tensor(doc[:remaining], dtype=torch.long)
                    pos += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        gpu_buffer.copy_(cpu_buffer, non_blocking=use_cuda)
        yield inputs, targets, epoch


def export_model(model, config, extra=None):
    """Write the trained model for the evaluator. Call once, at the end of training.

    Writes ``$RSI_EXPORT_DIR/state_dict.pt`` (tensors only; loaded with
    ``weights_only=True``) and ``config.json`` (passed to ``model.build_model(config)``
    by the evaluator). Unwraps ``torch.compile``. Returns the export dir.
    """
    os.makedirs(EXPORT_DIR, exist_ok=True)
    m = getattr(model, "_orig_mod", model)
    sd = {k: v.detach().to("cpu").contiguous() for k, v in m.state_dict().items()}
    tmp = os.path.join(EXPORT_DIR, "state_dict.pt.tmp")
    torch.save(sd, tmp)
    os.replace(tmp, os.path.join(EXPORT_DIR, "state_dict.pt"))
    with open(os.path.join(EXPORT_DIR, "config.json"), "w") as f:
        json.dump(config, f)
    meta = {"num_tensors": len(sd), "num_elements": int(sum(v.numel() for v in sd.values())),
            "extra": extra or {}}
    with open(os.path.join(EXPORT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f)
    print(f"[rsilib] exported {meta['num_tensors']} tensors ({meta['num_elements'] / 1e6:.1f}M elements) to {EXPORT_DIR}")
    return EXPORT_DIR
