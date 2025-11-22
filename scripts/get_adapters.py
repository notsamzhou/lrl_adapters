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
    full_sd = torch.load(args.checkpoint, map_location="cpu")
    missing, unexpected = pm.s2t_model.load_state_dict(full_sd, strict=False)

    encoder_adapter_sd = {}
    for k, v in pm.s2t_model.state_dict().items():
        k_l = k.lower()
        if not k.startswith("encoder."):
            continue
        if ("lora" in k_l) or ("adapter" in k_l):
            encoder_adapter_sd[k] = v.clone().cpu()

    total = sum(p.numel() for p in encoder_adapter_sd.values())
    print(f"Collected encoder only adapter tensors: {len(encoder_adapter_sd)} params, total elements: {total:,}")

    out_dir = args.out_dir or "adapters_export"
    os.makedirs(out_dir, exist_ok=True)
    out_name = args.out_name if args.out_name else f"encoder_only_{args.owsm_lang}.pth"
    out_path = os.path.join(out_dir, out_name)
    torch.save(encoder_adapter_sd, out_path)
    print(f"Saved adapter-only weights to: {out_path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="path to checkpoint, exp/adapter_por/<checkpoint>.pth")
    ap.add_argument("--adapter_config", default="config/finetune_adapter.yaml")
    ap.add_argument("--owsm_lang", required=True, help="source/target language code")
    ap.add_argument("--pretrained_model", default="espnet/owsm_v4_small_370M")
    ap.add_argument("--out_dir", default="adapters_export",
                    help="directory to save the filtered state_dict")
    ap.add_argument("--out_name", default=None,
                    help="optional filename for the exported .pth")
    args = ap.parse_args()
    main(args)
