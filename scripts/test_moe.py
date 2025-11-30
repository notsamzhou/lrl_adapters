#!/usr/bin/env python3

import os
import argparse

# Set cache directories BEFORE importing datasets/HF libraries
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
HF_CACHE_DIR = os.path.join(SCRIPT_DIR, '.cache', 'huggingface')
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(HF_CACHE_DIR, 'datasets')

import yaml
import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import jiwer
import datasets
from espnet2.bin.s2t_inference import Speech2Text
import espnetez as ez


class TokenLevelGating(nn.Module):
    """Token-level gating network - predicts expert weights per token."""
    def __init__(self, hidden_dim: int, num_experts: int, dropout: float = 0.1, training_noise: bool = True):
        super().__init__()
        self.num_experts = num_experts
        self.training_noise = training_noise
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_experts)
        )
        # Initialize for uniform distribution
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.gate(x)

        if self.training and self.training_noise:
            noise = torch.randn_like(logits) 
            logits = logits + noise

        top_k_vals, top_k_indices = torch.topk(logits, 2, dim=-1)
        

        top_k_weights = F.softmax(top_k_vals, dim=-1)
        top_k_weights = top_k_weights.to(dtype=logits.dtype)
        full_weights = torch.zeros_like(logits)
        full_weights.scatter_(-1, top_k_indices, top_k_weights)
        
        return full_weights

class MultiExpertLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, num_experts: int, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_linear = base_linear
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = alpha / rank
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.ParameterList([
            nn.Parameter(torch.zeros(rank, in_features), requires_grad=False)
            for _ in range(num_experts)
        ])
        self.lora_B = nn.ParameterList([
            nn.Parameter(torch.zeros(out_features, rank), requires_grad=False)
            for _ in range(num_experts)
        ])
        for param in self.base_linear.parameters():
            param.requires_grad = False
        self.current_gates = None

    def load_expert_weights(self, expert_idx: int, lora_A: torch.Tensor, lora_B: torch.Tensor):
        self.lora_A[expert_idx].data.copy_(lora_A)
        self.lora_B[expert_idx].data.copy_(lora_B)

    def set_gates(self, gates: torch.Tensor):
        self.current_gates = gates

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_linear(x)
        if self.current_gates is None:
            gates = torch.ones((*x.shape[:-1], self.num_experts), device=x.device, dtype=x.dtype) / self.num_experts
        else:
            gates = self.current_gates

        expert_outputs = []
        for i in range(self.num_experts):
            lora_out = x @ self.lora_A[i].T @ self.lora_B[i].T * self.scaling
            expert_outputs.append(lora_out)

        expert_outputs = torch.stack(expert_outputs, dim=0)
        gates_expanded = gates.permute(-1, *range(len(gates.shape)-1)).unsqueeze(-1)
        
        # Safe broadcasting
        if gates_expanded.shape[2] != expert_outputs.shape[2]:
             # In eval mode, we might want to handle this gracefully or error out
             pass 

        lora_delta = (gates_expanded * expert_outputs).sum(dim=0)
        return result + lora_delta

class EncoderMoEPreHook:
    def __init__(self, gating_module, layer_modules):
        self.gating_module = gating_module
        self.layer_modules = layer_modules
    def __call__(self, module, args):
        x = args[0] if isinstance(args, tuple) else args
        gates = self.gating_module(x)
        for m in self.layer_modules:
            m.set_gates(gates)
        return None

class DecoderMoEPreHook:
    def __init__(self, gating_net_x, gating_net_mem, std_modules, mem_modules):
        self.gating_net_x = gating_net_x
        self.gating_net_mem = gating_net_mem
        self.std_modules = std_modules
        self.mem_modules = mem_modules
    def __call__(self, module, args):
        if isinstance(args, tuple) and len(args) >= 3:
            x = args[0]
            memory = args[2]
        else:
            return None
        
        gates_x = self.gating_net_x(x)
        for m in self.std_modules:
            m.set_gates(gates_x)

        if self.mem_modules and memory is not None:
            gates_mem = self.gating_net_mem(memory)
            for m in self.mem_modules:
                m.set_gates(gates_mem)
        return None


def load_adapter_checkpoint(path: str, prefix: str = "") -> dict:
    sd = torch.load(path, map_location='cpu')
    if prefix:
        filtered = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                filtered[k[len(prefix):]] = v
        return filtered
    return sd

def replace_linear_with_multi_expert(module, target_modules, num_experts, rank, alpha, expert_params):
    replaced = {}
    def replace_in_module(parent, parent_name=""):
        for name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{name}" if parent_name else name
            if isinstance(child, MultiExpertLoRALinear):
                replaced[full_name] = child
                continue
            is_target = any(full_name.endswith(t) for t in target_modules)
            if is_target and isinstance(child, nn.Linear):
                multi_expert = MultiExpertLoRALinear(child, num_experts, rank, alpha)
                loaded = 0
                for i, params in enumerate(expert_params):
                    key_A = f"{full_name}.lora_A"
                    key_B = f"{full_name}.lora_B"
                    if key_A in params and key_B in params:
                        multi_expert.load_expert_weights(i, params[key_A], params[key_B])
                        loaded += 1
                if loaded > 0:
                    setattr(parent, name, multi_expert)
                    replaced[full_name] = multi_expert
            replace_in_module(child, full_name)
    replace_in_module(module)
    return replaced

def rebuild_moe_architecture(model, adapter_config, adapter_paths):
    """Reconstructs the MoE architecture exactly as done in training."""
    target_modules = adapter_config['adapter_conf']['target_modules']
    rank = adapter_config['adapter_conf']['rank']
    alpha = adapter_config['adapter_conf']['alpha']
    num_experts = len(adapter_paths)
    
    enc_dim = model.encoder.encoders[0].size
    dec_dim = model.decoder.decoders[0].size

    # Load checkpoints for structure initialization
    enc_params = [load_adapter_checkpoint(p, "encoder.") for p in adapter_paths]
    dec_params = [load_adapter_checkpoint(p, "decoder.") for p in adapter_paths]

    # Encoder Setup
    enc_replaced = replace_linear_with_multi_expert(
        model.encoder, target_modules, num_experts, rank, alpha, enc_params
    )
    enc_layer_modules = [[] for _ in range(len(model.encoder.encoders))]
    for name, mod in enc_replaced.items():
        if "encoders." in name:
            idx = int(name.split("encoders.")[1].split('.')[0])
            enc_layer_modules[idx].append(mod)
            
    enc_gates = nn.ModuleList([
        TokenLevelGating(enc_dim, num_experts) for _ in range(len(model.encoder.encoders))
    ])

    # Decoder Setup
    dec_replaced = replace_linear_with_multi_expert(
        model.decoder, target_modules, num_experts, rank, alpha, dec_params
    )
    dec_std_modules = [[] for _ in range(len(model.decoder.decoders))]
    dec_mem_modules = [[] for _ in range(len(model.decoder.decoders))]

    for name, mod in dec_replaced.items():
        if "decoders." in name:
            idx = int(name.split("decoders.")[1].split('.')[0])
            is_cross_kv = ("src_attn" in name) and (name.endswith("linear_k") or name.endswith("linear_v"))
            if is_cross_kv:
                dec_mem_modules[idx].append(mod)
            else:
                dec_std_modules[idx].append(mod)

    dec_gates_x = nn.ModuleList([
        TokenLevelGating(dec_dim, num_experts) for _ in range(len(model.decoder.decoders))
    ])
    dec_gates_mem = nn.ModuleList([
        TokenLevelGating(enc_dim, num_experts) for _ in range(len(model.decoder.decoders))
    ])

    return enc_gates, enc_layer_modules, dec_gates_x, dec_gates_mem, dec_std_modules, dec_mem_modules


def cyrillic_to_latin_serbian(text):
    """Convert Serbian Cyrillic to Latin script"""
    cyrillic_to_latin = {
        # Core Serbian Cyrillic alphabet
        'А': 'A', 'а': 'a',
        'Б': 'B', 'б': 'b',
        'В': 'V', 'в': 'v',
        'Г': 'G', 'г': 'g',
        'Д': 'D', 'д': 'd',
        'Ђ': 'Đ', 'ђ': 'đ',
        'Е': 'E', 'е': 'e',
        'Ж': 'Ž', 'ж': 'ž',
        'З': 'Z', 'з': 'z',
        'И': 'I', 'и': 'i',
        'Ј': 'J', 'ј': 'j',
        'К': 'K', 'к': 'k',
        'Л': 'L', 'л': 'l',
        'Љ': 'Lj', 'љ': 'lj',
        'М': 'M', 'м': 'm',
        'Н': 'N', 'н': 'n',
        'Њ': 'Nj', 'њ': 'nj',
        'О': 'O', 'о': 'o',
        'П': 'P', 'п': 'p',
        'Р': 'R', 'р': 'r',
        'С': 'S', 'с': 's',
        'Т': 'T', 'т': 't',
        'Ћ': 'Ć', 'ћ': 'ć',
        'У': 'U', 'у': 'u',
        'Ф': 'F', 'ф': 'f',
        'Х': 'H', 'х': 'h',
        'Ц': 'C', 'ц': 'c',
        'Ч': 'Č', 'ч': 'č',
        'Џ': 'Dž', 'џ': 'dž',
        'Ш': 'Š', 'ш': 'š',
        # Additional Russian/borrowed Cyrillic letters (used in loanwords)
        'Ё': 'Ë', 'ё': 'ë',
        'Й': 'J', 'й': 'j',
        'Щ': 'Šč', 'щ': 'šč',
        'Ъ': '', 'ъ': '',  # hard sign - usually omitted
        'Ы': 'Y', 'ы': 'y',
        'Ь': '', 'ь': '',  # soft sign - usually omitted
        'Э': 'E', 'э': 'e',
        'Ю': 'Ju', 'ю': 'ju',
        'Я': 'Ja', 'я': 'ja',
    }
    result = []
    for char in text:
        result.append(cyrillic_to_latin.get(char, char))
    return ''.join(result)

def normalize_text(text, owsm_lang=None):
    if owsm_lang == 'srp':
        text = cyrillic_to_latin_serbian(text)
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_finetuned_moe_model(checkpoint_path, test_owsm_lang, adapter_config_path, adapter_paths, device='cuda'):
    print(f"Loading base model with prompt <{test_owsm_lang}>...")
    pretrained_model = Speech2Text.from_pretrained(
        "espnet/owsm_v4_small_370M",
        lang_sym=f"<{test_owsm_lang}>",
        beam_size=5,
        device='cuda' 
    )
    model = pretrained_model.s2t_model

    with open(adapter_config_path, 'r') as f:
        adapter_config = yaml.safe_load(f)


    print("Rebuilding MoE architecture:")
    (enc_gates, enc_mods, 
     dec_gates_x, dec_gates_mem, 
     dec_std_mods, dec_mem_mods) = rebuild_moe_architecture(model, adapter_config, adapter_paths)

    model.moe_components = nn.ModuleDict({
        'enc_gates': enc_gates,
        'dec_gates_x': dec_gates_x,
        'dec_gates_mem': dec_gates_mem,
    })

    print("Registering Hooks:")
    for i, layer in enumerate(model.encoder.encoders):
        if enc_mods[i]:
            hook = EncoderMoEPreHook(model.moe_components['enc_gates'][i], enc_mods[i])
            layer.register_forward_pre_hook(hook)

    for i, layer in enumerate(model.decoder.decoders):
        if dec_std_mods[i] or dec_mem_mods[i]:
            hook = DecoderMoEPreHook(
                model.moe_components['dec_gates_x'][i],
                model.moe_components['dec_gates_mem'][i],
                dec_std_mods[i],
                dec_mem_mods[i]
            )
            layer.register_forward_pre_hook(hook)


    print(f"Loading checkpoint: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location='cpu')

    if list(state_dict.keys())[0].startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Strict=True ensures our architecture matches the checkpoint
    model.load_state_dict(state_dict, strict=True) 
    
    model.to(device)
    model.eval()
    
    pretrained_model.s2t_model = model
    pretrained_model.device = device
    return pretrained_model

def main(args):
    # IMPORTANT: These paths must exist and match what was used in training
    # to initialize the expert shapes correctly.
    ADAPTER_PATHS = [
        'exp/adapter_cym/adapters_only_cym.pt',
        'exp/adapter_rus/adapters_only_rus.pt',
        'exp/adapter_ita/adapters_only_ita.pt',
        'exp/adapter_por/adapters_only_por.pt',
        'exp/adapter_hun/adapters_only_hun.pt',
    ]

    model_pipeline = load_finetuned_moe_model(
        args.checkpoint,
        args.owsm_lang,
        args.adapter_config,
        ADAPTER_PATHS,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    print(f"Loading dataset: {args.fleurs_lang}")
    try:
        fleurs_hf = datasets.load_dataset("google/fleurs", args.fleurs_lang, trust_remote_code=True, split='test')
    except:
        fleurs_hf = datasets.load_dataset("google/fleurs", args.fleurs_lang, trust_remote_code=True, download_mode="force_redownload", split='test')

    if args.num_samples:
        fleurs_hf = fleurs_hf.select(range(min(args.num_samples, len(fleurs_hf))))

    references = []
    hypotheses = []

    print(f"Starting inference on {len(fleurs_hf)} samples...")

    for i, sample in enumerate(fleurs_hf):
        audio = sample['audio']['array'].astype(np.float32)
        ref = sample['transcription']

        with torch.no_grad():
            # Speech2Text pipeline handles normalization/tokenization internally
            # It returns a list of results, we take the top beam
            results = model_pipeline(audio)
            hyp = results[0][0]


        if hyp.startswith("<"):
             last_bracket = hyp.rfind('>')
             if last_bracket != -1:
                 hyp = hyp[last_bracket + 1:].strip()

        # Normalize
        ref_norm = normalize_text(ref, args.owsm_lang)
        hyp_norm = normalize_text(hyp, args.owsm_lang)

        references.append(ref_norm)
        hypotheses.append(hyp_norm)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1} samples...")

    wer = jiwer.wer(references, hypotheses)
    cer = jiwer.cer(references, hypotheses)

    print(f"\nFinal Results for {args.fleurs_lang}:")
    print(f"WER: {wer:.4f}")
    print(f"CER: {cer:.4f}")

    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Checkpoint: {args.checkpoint}\n")
            f.write(f"Lang: {args.fleurs_lang}\n")
            f.write(f"WER: {wer:.4f}\nCER: {cer:.4f}\n\n")
            for r, h in zip(references, hypotheses):
                f.write(f"REF: {r}\nHYP: {h}\n\n")
        print(f"Saved details to {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fleurs_lang", required=True)
    parser.add_argument("--owsm_lang", required=True)
    parser.add_argument("--adapter_config", default="config/finetune_adapter.yaml")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    main(parser.parse_args())