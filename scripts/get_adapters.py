#!/usr/bin/env python3
import os, argparse, torch, yaml
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.layers.create_adapter import create_adapter

def is_lora_param(name):
    # guessing from name if param is from lora/adapter
    ln = name.lower()
    return ("lora" in ln) or ("adapter" in ln)

def main(args):
    print(f"Loading base: {args.pretrained_model}")
    pm = Speech2Text.from_pretrained(args.pretrained_model, lang_sym=f"<{args.owsm_lang}>",
                                     beam_size=1, device="cpu")
    with open(args.adapter_config, "r") as f:
        cfg = yaml.safe_load(f)
    create_adapter(pm.s2t_model, cfg["adapter"], cfg["adapter_conf"])

    print(f"Loading trained state dict: {args.checkpoint}")
    sd = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = pm.s2t_model.load_state_dict(sd, strict=False)

    adapters_only = {}
    for n, p in pm.s2t_model.named_parameters():
        if is_lora_param(n):
            adapters_only[n] = p.detach().cpu()
    for n, b in pm.s2t_model.named_buffers():
        if is_lora_param(n):
            adapters_only[n] = b.detach().cpu()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    torch.save(adapters_only, args.output)
    print(f"Saved adapter-only weights to: {args.output}")
    print(f"Count keys: {len(adapters_only)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to checkpoint, exp/adapter_por/<checkpoint>.pth")
    ap.add_argument("--adapter_config", default="config/finetune_adapter.yaml")
    ap.add_argument("--owsm_lang", required=True, help="source/target language code")
    ap.add_argument("--pretrained_model", default="espnet/owsm_v4_small_370M")
    ap.add_argument("--output", required=True,
                    help="Where to save adapter-only file, e.g. exp/adapter_por/adapters_only.pt")
    args = ap.parse_args()
    main(args)
