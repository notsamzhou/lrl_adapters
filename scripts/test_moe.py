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
    """Dense Gating - Uses weighted average of ALL experts."""
    def __init__(self, hidden_dim: int, num_experts: int, dropout: float = 0.1, training_noise: bool = True, target_lang: str = None, mean_pooling=False):
        super().__init__()
        self.num_experts = num_experts
        self.training_noise = training_noise
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, num_experts)
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.temperature = 0.4
        self.mean_pooling = mean_pooling

        nn.init.kaiming_uniform_(self.gate[-1].weight)
        with torch.no_grad():
            self.gate[-1].weight.mul_(0.01)

        nn.init.zeros_(self.gate[-1].bias)
        if target_lang:
            prior_tensor = torch.tensor(expert_priors(target_lang), dtype=self.gate[-1].bias.dtype)
            self.gate[-1].bias.data.copy_(prior_tensor)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if self.mean_pooling:
            x = x.mean(dim=1)

        x = self.layer_norm(x)
        logits = self.gate(x)

        # Randomly eliminate base expert
        if self.training:
             if torch.rand(1).item() < 0.15:
                 logits[..., -1] = -1e9

        if self.training and self.training_noise:
            # Noise helps prevent the router from collapsing to just one expert early on
            noise = torch.randn_like(logits) * 0.1 
            logits = logits + noise

        weights = F.softmax(logits / self.temperature, dim=-1)


        if self.mean_pooling:
            weights = weights.unsqueeze(1)
        
        return weights

def expert_priors(target_lang: str):
    # cym , rus, ita, por, hun, base
    priors = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    if target_lang == 'srp':
        priors = [-1.0,  2.0,  -1.0, -1.0, -1.0, 1.0]
    if target_lang == 'hye':
        priors = [-1.0,  -1.0, -1.0, -1.0, -1.0, 1.0]
    if target_lang == 'gle':
        priors = [ 2.0, -1.0, -1.0, -1.0, -1.0, 1.0]
    if target_lang == 'nob':
        priors = [-1.0,  -1.0,  -1.0, -1.0, -1.0, 1.0]
    if target_lang == 'oci':
        priors = [-1.0, -1.0,  2.0,  2.0, -1.0, 1.0]
    return priors



class MultiExpertLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, num_experts: int, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base_linear = base_linear
        self.num_experts = num_experts
        self.rank = rank
        self.scaling = alpha / rank
        in_features = base_linear.in_features
        out_features = base_linear.out_features

        self.lora_A = nn.Parameter(torch.stack([
            torch.zeros(rank, in_features) for _ in range(num_experts)
        ]))
        self.lora_B = nn.Parameter(torch.stack([
            torch.zeros(out_features, rank) for _ in range(num_experts)
        ]))
        
        # Freeze base, Freeze experts
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

        gates = self.current_gates
        
        # Safety Check for Inference Crash
        if gates is not None:
            if gates.shape[0] != x.shape[0]:
                gates = None 
            elif gates.shape[1] != x.shape[1] and gates.shape[1] != 1:
                gates = None
        
        if gates is None:
            # Fallback to Uniform Soft weights
            gates = torch.ones(
                (*x.shape[:-1], self.num_experts),
                device=x.device, dtype=x.dtype
            ) / self.num_experts

        # x shape: Batch, Seq, In
        # A shape: Experts, Rank, In
        # B shape: Experts, Out, Rank
        x_a = torch.einsum('bsi, eri -> bser', x, self.lora_A)
        expert_outputs = torch.einsum('bser, eor -> bseo', x_a, self.lora_B)

        # Gates: [Batch, Seq, NumExperts] -> [Batch, Seq, NumExperts, 1]
        gates_expanded = gates.unsqueeze(-1)
        
        lora_delta = (gates_expanded * expert_outputs).sum(dim=2)

        max_gate_weight = gates.max(dim=-1, keepdim=True).values
        scaling_factor = 1.0 / (max_gate_weight + 1e-6)
        lora_delta = lora_delta * scaling_factor
        
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
        x = None
        memory = None

        if isinstance(args, tuple):
            if len(args) >= 1:
                x = args[0]
            if len(args) >= 3:
                memory = args[2]
        
        # 1. Gate on Decoder Input (tgt)
        if x is not None:
            gates_x = self.gating_net_x(x)
            for m in self.std_modules:
                m.set_gates(gates_x)
        
        # 2. Gate on Memory (Encoder Output) - Only if memory modules exist
        if self.mem_modules:
            if memory is not None:
                gates_mem = self.gating_net_mem(memory)
                for m in self.mem_modules:
                    m.set_gates(gates_mem)
            else:
                # IMPORTANT: Clear memory gates if memory is not present
                # so MultiExpertLoRALinear falls back to uniform instead of using stale gates.
                for m in self.mem_modules:
                    m.set_gates(None)

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
                
                # Load weights
                loaded_count = 0
                for i, params in enumerate(expert_params):
                    # Skip the last expert if it's the empty "Base" expert
                    if not params: 
                        continue 

                    key_A = f"{full_name}.lora_A"
                    key_B = f"{full_name}.lora_B"
                    
                    multi_expert.load_expert_weights(i, params[key_A], params[key_B])
                    loaded_count += 1


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
        model.encoder, target_modules, num_experts+1, rank, alpha, enc_params
    )
    enc_layer_modules = [[] for _ in range(len(model.encoder.encoders))]
    for name, mod in enc_replaced.items():
        if "encoders." in name:
            idx = int(name.split("encoders.")[1].split('.')[0])
            enc_layer_modules[idx].append(mod)
            
    enc_gates = nn.ModuleList([
        TokenLevelGating(enc_dim, num_experts+1, mean_pooling=False) for _ in range(len(model.encoder.encoders))
    ])

    # Decoder Setup
    dec_replaced = replace_linear_with_multi_expert(
        model.decoder, target_modules, num_experts+1, rank, alpha, dec_params
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
        TokenLevelGating(dec_dim, num_experts+1, mean_pooling=False) for _ in range(len(model.decoder.decoders))
    ])
    dec_gates_mem = nn.ModuleList([
        TokenLevelGating(enc_dim, num_experts+1, mean_pooling=False) for _ in range(len(model.decoder.decoders))
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
            f.write(f"Checkpoint Evaluation Results\n")
            f.write(f"Model checkpoint: {args.checkpoint}\n")
            f.write(f"Testing on: {args.fleurs_lang} (OWSM: {args.owsm_lang})\n")
            f.write(f"Samples: {len(references)}\n")
            f.write(f"WER: {wer:.4f} ({wer*100:.2f}%)\n")
            f.write(f"CER: {cer:.4f} ({cer*100:.2f}%)\n\n")

            for i, (ref, hyp) in enumerate(zip(references[:5], hypotheses[:5])):
                f.write(f"Sample {i+1}:\n")
                f.write(f"  REF: {ref}\n")
                f.write(f"  HYP: {hyp}\n\n")
        print(f"\nResults saved to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--fleurs_lang", required=True)
    parser.add_argument("--owsm_lang", required=True)
    parser.add_argument("--adapter_config", default="config/finetune_adapter.yaml")
    parser.add_argument("--output_file", default=None)
    parser.add_argument("--num_samples", type=int, default=None)
    main(parser.parse_args())