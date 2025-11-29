# utils/adapter_io.py
import torch
from typing import Dict, List, Tuple

def _get_encoder_blocks(encoder: torch.nn.Module):
    for attr in ["encoders", "layers", "blocks"]:
        if hasattr(encoder, attr):
            seq = getattr(encoder, attr)
            try:
                return list(seq)
            except Exception:
                pass
    return [encoder]

@torch.no_grad()
def collect_encoder_layer_outputs(
    model,                     # full ESPnet S2T model (has frontend + encoder)
    wavs: torch.Tensor,        # (B, T) float32 waveform
    wav_lens: torch.Tensor,    # (B,) lengths in SAMPLES
    layer_ids: List[int],      # e.g. [-1, 0, 4, 8, 12, 16, 20]
) -> Tuple[Dict[int, torch.Tensor], torch.Tensor]:
    """
    Runs FRONTEND -> ENCODER with hooks and returns per-layer activations.
    All layer_ids outside range are ignored. -1 maps to the final encoder output.
    Returns:
      layer_outs: {layer_id: (B, T', D)}
      hlens: (B,) encoder-lengths in frames (or None if unavailable)
    """
    enc = model.encoder
    device = wavs.device

    # ---- extract features FIRST (avoids quadratic masks in samples) ----
    feats, feat_lens = model._extract_feats(wavs, wav_lens)
    if isinstance(feats, (tuple, list)):
        feats = feats[0]

    # ---- set up hooks ----
    blocks = _get_encoder_blocks(enc)
    max_idx = len(blocks) - 1
    want = sorted(set([i for i in layer_ids if (i == -1 or 0 <= i <= max_idx)])) or [-1]

    storage: Dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(idx: int):
        def _hook(_m, _inp, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            storage[idx] = x.detach()
        return _hook

    for i in want:
        if i == -1:  # final output captured after forward
            continue
        h = blocks[i].register_forward_hook(make_hook(i))
        handles.append(h)

    # ---- encoder forward ----
    out = enc(feats, feat_lens)
    if isinstance(out, (tuple, list)):
        hs = out[0]                                      # (B, T', D)
        hlens = out[1] if len(out) > 1 else None         # (B,) or None
    else:
        hs, hlens = out, None

    storage[-1] = hs

    if hlens is None:
        hlens = torch.full(
            (hs.size(0),), hs.size(1),
            dtype=torch.long, device=hs.device
        )

    for h in handles:
        try: h.remove()
        except Exception: pass

    layer_outs = {l: storage[l] for l in want if l in storage}
    if not layer_outs:   # safety fallback
        layer_outs = {-1: hs}

    return layer_outs, hlens

def load_adapter_into_model_debug(model, adapter_state, tag=""):
    """Load with stats; returns (matched, unexpected, missing)."""
    model_sd = model.state_dict()

    could_match = sum(1 for k in adapter_state.keys() if k in model_sd)

    # Do the load (non-strict)
    missing, unexpected = model.load_state_dict(adapter_state, strict=False)

    matched = len(adapter_state) - len(unexpected)
    print(f"[adapter{(':'+tag) if tag else ''}] keys_in_file={len(adapter_state)} "
          f"could_match={could_match} matched={matched} missing={len(missing)} unexpected={len(unexpected)}")
    if matched == 0:
        # Show a few examples to compare name patterns
        print("[adapter] example adapter keys:", list(adapter_state.keys())[:5])
        print("[adapter] example model adapter keys:",
              [k for k in model_sd.keys() if 'adapter' in k][:5])
    return matched, unexpected, missing