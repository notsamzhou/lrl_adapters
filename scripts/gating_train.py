# scripts/gating_train.py
"""
Gate training over encoder-only source adapters using forward hooks (frontend-safe).

Pipeline per batch:
  - For each source adapter:
      * swap adapter weights into model
      * collect per-layer encoder outputs via collect_encoder_layer_outputs()
        (uses model.frontend / _extract_feats internally)
  - For each gated layer:
      * mean-pool (B,T,D) -> (B,D) for scoring ONLY
      * gate produces (B,S) logits; top-k masked softmax -> (B,S) weights
      * fuse full (B,T,D) time series across sources using those weights
  - Take the largest requested/available layer as encoder-out; compute CTC loss with hlens

Saves to:   <save_dir>/<ow sm tags joined>/gate_epoch{N}.pt
"""

import os
import argparse
import yaml
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import datasets
import numpy as np

from espnet2.bin.s2t_inference import Speech2Text
from espnet2.layers.create_adapter import create_adapter

from utils.adapter_io import collect_encoder_layer_outputs

# ---------------- Gate head ----------------

class SimpleLayerGate(nn.Module):
    """
    Per-layer, content-conditioned scorer.
    For each layer l, has a learnable w_l in R^D (and bias) that scores each source's (B,D) pooled rep.
    """
    def __init__(self, d_model: int, layer_ids: List[int], num_sources: int):
        super().__init__()
        self.d_model = d_model
        self.layer_ids = list(layer_ids)
        self.w = nn.ParameterDict({str(l): nn.Parameter(torch.zeros(d_model)) for l in self.layer_ids})
        self.b = nn.ParameterDict({str(l): nn.Parameter(torch.zeros(1)) for l in self.layer_ids})
        for l in self.layer_ids:
            nn.init.normal_(self.w[str(l)], mean=0.0, std=0.02)

    def forward(self, pooled_by_layer: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
        """
        pooled_by_layer[l]: (B, S, D)
        returns logits[l]: (B, S)
        """
        logits = {}
        for l, H in pooled_by_layer.items():
            w = self.w[str(l)]  # (D,)
            b = self.b[str(l)]  # (1,)
            scores = torch.einsum("bsd,d->bs", H, w) + b
            logits[l] = scores
        return logits

class MlpLayerGate(nn.Module):
    def __init__(self, d_model: int, layer_ids, hidden=512, dropout=0.1):
        super().__init__()
        self.layer_ids = list(layer_ids)
        self.mlps = nn.ModuleDict()
        for l in self.layer_ids:
            self.mlps[str(l)] = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    def forward(self, pooled_by_layer):
        logits = {}
        for l, H in pooled_by_layer.items():      # H: (B,S,D)
            B,S,D = H.shape
            x = H.reshape(B*S, D)
            s = self.mlps[str(l)](x).reshape(B, S)  # (B,S)
            logits[l] = s
        return logits


def topk_masked_softmax(logits: torch.Tensor, k: int, tau: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
    topk_val, topk_idx = torch.topk(logits, k=min(k, logits.size(1)), dim=-1)
    masked = torch.full_like(logits, float("-inf"))
    masked.scatter_(dim=-1, index=topk_idx, src=topk_val)
    probs = torch.softmax(masked / max(tau, 1e-6), dim=-1)
    return probs, topk_idx


# ---------------- Data ----------------

def build_datasets(fleurs_codes: List[str], lang_tags: List[str], split="train"):
    """
    Returns list of dicts with 'audio' (np.float32), 'lang', 'text'.
    Skips clips < 0.3 s to avoid frontend/conv edge cases.
    """
    merged = []
    for fleurs_lang, owsm_lang in zip(fleurs_codes, lang_tags):
        ds = datasets.load_dataset("google/fleurs", fleurs_lang, trust_remote_code=True)[split]
        for ex in ds:
            wav = ex["audio"]["array"]
            if wav is None:
                continue
            if len(wav) < int(0.3 * 16000):
                continue
            merged.append({
                "audio": wav.astype(np.float32),
                "lang": owsm_lang,
                "text": ex["transcription"],
            })
    rng = np.random.default_rng(42)
    rng.shuffle(merged)
    return merged


def collate(items, tokenizer, converter, device, pad_id, max_seconds=None):
    """
    Raw waveforms only; features are extracted inside collect_encoder_layer_outputs().
    """
    wavs = [torch.from_numpy(x["audio"]) for x in items]
    if max_seconds is not None:
        max_samples = int(16000 * max_seconds)
        wavs = [w[:max_samples] for w in wavs]

    xlens = torch.tensor([w.shape[0] for w in wavs], dtype=torch.long)
    Tmax = int(xlens.max().item())
    # light floor to avoid degenerate shapes
    Tmax = max(Tmax, 160)
    xs = torch.zeros(len(wavs), Tmax, dtype=torch.float32)
    for i, w in enumerate(wavs):
        L = min(w.shape[0], Tmax)
        xs[i, :L] = w[:L]

    # OWSM prompts for decoder tokenization but since we are doing ctc loss only for now, remove tokens at beginning
    IGNORE_ID = -1
    ys_list = []
    for it in items:
        # prompt = f"<{it['lang']}><asr><notimestamps> {it['text']}"
        ids = converter.tokens2ids(tokenizer.text2tokens(it['text'])) #transcription only, can change to prompt if doing decoder stuff
        ys_list.append(torch.tensor(ids, dtype=torch.long))

    ylens = torch.tensor([t.numel() for t in ys_list], dtype=torch.long)
    Lmax = int(ylens.max().item())
    ys = torch.full((len(ys_list), Lmax), fill_value=IGNORE_ID, dtype=torch.long)
    for i, t in enumerate(ys_list):
        ys[i, :t.numel()] = t

    return xs.to(device), xlens.to(device), ys.to(device), ylens.to(device)


# ---------------- Adapter swap ----------------

def load_adapter_into_model(model: nn.Module, adapter_state: Dict[str, torch.Tensor]):
    # Non-strict so only adapter keys map; base model stays frozen
    model.load_state_dict(adapter_state, strict=False)


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pretrained_model", default="espnet/owsm_v4_small_370M")
    ap.add_argument("--target_fleurs", nargs="+", required=True, help="e.g. ga_ie oc_fr sr_rs nb_no hy_am")
    ap.add_argument("--target_owsm",   nargs="+", required=True, help="e.g. gle oci srp nob hye")
    ap.add_argument("--adapter_bank",  nargs="+", required=True, help="paths to exported source adapters .pt")
    ap.add_argument("--layers", default="0,4,8,12,16,20", help="encoder block indices to gate; -1 always captured")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--tau",  type=float, default=1.0)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--save_dir", default="exp/gate_target")
    ap.add_argument("--max_seconds", type=float, default=12.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.save_dir, exist_ok=True)

    # Subdir per target set to avoid collisions
    target_tag = "+".join(args.target_owsm)
    run_dir = os.path.join(args.save_dir, target_tag)
    os.makedirs(run_dir, exist_ok=True)

    # 1) Load base model and create empty adapter slots
    pm = Speech2Text.from_pretrained(
        args.pretrained_model,
        lang_sym=f"<{args.target_owsm[0]}>",
        beam_size=1,
        device=device,
    )
    model = pm.s2t_model
    with open("config/finetune_adapter.yaml") as f:
        cfg = yaml.safe_load(f)
    create_adapter(model.encoder, cfg["adapter"], cfg["adapter_conf"])
    for _, m in model.named_modules():
        if hasattr(m, "merge_weights"):
            m.merge_weights = False
        if hasattr(m, "merged"):
            m.merged = False

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # 2) Load adapter tensors (CPU RAM)
    adapter_states = [torch.load(pth, map_location="cpu") for pth in args.adapter_bank]

    # 3) Data
    train_set = build_datasets(args.target_fleurs, args.target_owsm, split="train")
    tokenizer = pm.tokenizer
    converter = pm.converter
    token_list = pm.s2t_train_args.token_list
    pad_id = token_list.index("<pad>") if "<pad>" in token_list else 0
    loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda items: collate(items, tokenizer, converter, device, pad_id, max_seconds=args.max_seconds),
    )

    # 4) Determine available layers from a real batch
    #    (intersect requested with those keys returned by collect)
    req_layers = [int(x) for x in args.layers.split(",") if x.strip()]
    req_layers = sorted(set(req_layers + [-1]))  # always include -1
    with torch.no_grad():
        xs_probe = torch.zeros(1, 16000, dtype=torch.float32, device=device)
        xl_probe = torch.tensor([16000], dtype=torch.long, device=device)

        req_layers = sorted(set([-1] + [int(x) for x in args.layers.split(",") if x.strip()]))

        probe_outs, _ = collect_encoder_layer_outputs(model, xs_probe, xl_probe, layer_ids=req_layers)
        available = sorted(probe_outs.keys())
        # infer D from final
        final_key = -1 if -1 in probe_outs else max(available)
        D = probe_outs[final_key].shape[-1]
    # Final layer set (always include -1 if the encoder exposes it)
    gated_layers = sorted(set([l for l in req_layers if l in available]))
    if -1 in available and -1 not in gated_layers:
        gated_layers.append(-1)
    gated_layers = sorted(gated_layers)

    # The fused encoder output will come from the largest real layer (else -1)
    nonneg = [l for l in gated_layers if l >= 0]
    last_l = max(nonneg) if nonneg else -1

    print(f"[gate] requested layers: {req_layers}")
    print(f"[gate] available layers: {available}")
    print(f"[gate] using layers    : {gated_layers}")

    # infer d_model from the probe outputs
    any_key = max(gated_layers)  # just pick a valid layer (pref last)
    D = probe_outs[any_key].shape[-1]

    # 5) Build the gate
    gate_layers = [l for l in gated_layers if l != -1]  # we don't gate over "-1"
    # gate = SimpleLayerGate(d_model=D, layer_ids=gate_layers, num_sources=S).to(device)
    gate = MlpLayerGate(d_model=D, layer_ids=gate_layers, hidden=512).to(device)
    opt = optim.AdamW(gate.parameters(), lr=args.lr)

    # 6) Train
    for epoch in range(1, args.epochs + 1):
        gate.train()
        running = 0.0

        for step, (xs, xlens, ys, ylens) in enumerate(loader):
            # Collect (B,T,D) for each source adapter and each gated layer

            #CHANGED:
            per_layer_per_source = {l: [] for l in gated_layers if l != -1}
            hlens_ref = None

            # we should be calculating CTC loss using final layer, not last encoder layer which is still intermediate
            per_source_final = []

            for astate in adapter_states:
                load_adapter_into_model(model, astate)
                with torch.no_grad():
                    # run collector ON WAVEFORMS (it will handle frontend+encoder+hooks)
                    layer_outs, hlens = collect_encoder_layer_outputs(model, xs, xlens, layer_ids=gated_layers)
                if hlens_ref is None:
                    hlens_ref = hlens

                per_source_final.append(layer_outs[-1].to(device))

                # mean-pool time -> (B,D) and keep only real (non -1) layers
                for l, x in layer_outs.items():
                    if l == -1:
                        continue
                    per_layer_per_source[l].append(x.to(device))

            # For scoring: time-mean pool → (B,D), then stack to (B,S,D)
            pooled_by_layer: Dict[int, torch.Tensor] = {}
            for l, lst in per_layer_per_source.items():
                # lst: length S of (B,T,D)
                pooled = [h.mean(dim=1) for h in lst]            # S × (B,D)
                H_s_bd = torch.stack(pooled, dim=0).transpose(0, 1).contiguous()  # (B,S,D)
                pooled_by_layer[l] = H_s_bd.to(device)

            # Gate logits per layer → weights
            logits_by_layer = gate(pooled_by_layer)  # l -> (B,S)

            #using last actual layer
            logits_last = logits_by_layer[last_l]
            w_bs, _ = topk_masked_softmax(logits_last, k=args.topk, tau=args.tau)
            H_final = torch.stack(per_source_final, dim=1)
            fused_final = (w_bs[:, :, None, None] * H_final).sum(dim=1)
            loss = model.ctc(fused_final, hlens_ref, ys, ylens)

            #ignore for now
            # fused_by_layer: Dict[int, torch.Tensor] = {}

            # for l, lst in per_layer_per_source.items():
            #     # stack full time series for fusion: (S,B,T,D) -> (B,S,T,D)
            #     H_s = torch.stack(lst, dim=0).transpose(0, 1).contiguous().to(device)  # (B,S,T,D)
            #     logits = logits_by_layer[l]                                              # (B,S)
            #     w_bs, _ = topk_masked_softmax(logits, k=args.topk, tau=args.tau)        # (B,S)
            #     fused = (w_bs[:, :, None, None] * H_s).sum(dim=1)                       # (B,T,D)
            #     fused_by_layer[l] = fused

            # # Choose fused encoder output from last_l
            # fused_encoder_out = fused_by_layer[last_l]  # (B,T,D)
            # enc_lens = hlens_ref                        # (B,)


            # CTC loss (CTC-only here; AED disabled) #right now, only calculating loss from last fused layer
            # maybe eventually add auxiliary losses at intermediate layers with small CTC heads per layer so that those gates can learn something
            # loss = model.ctc(fused_encoder_out, enc_lens, ys, ylens)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += float(loss.item())
            if (step + 1) % 20 == 0:
                print(f"[epoch {epoch}] step {step+1} loss {running/20:.4f}")
                running = 0.0

        # Save per-epoch into the per-target subdir
        S = len(adapter_states)
        ckpt = {
            "state_dict": gate.state_dict(),
            "layers": gate_layers,
            "num_sources": S,
            "topk": args.topk,
            "tau": args.tau,
            "adapter_bank": args.adapter_bank,
            "d_model": D,
            "targets": args.target_owsm,
        }
        path = os.path.join(run_dir, f"gate_epoch{epoch}.pt")
        torch.save(ckpt, path)
        print(f"[epoch {epoch}] saved {path}")


if __name__ == "__main__":
    main()
