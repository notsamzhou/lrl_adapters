#!/usr/bin/env python3

import os
import argparse
from typing import Dict, List, Tuple

# Set cache directories BEFORE importing datasets/HF libraries
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(SCRIPT_DIR, '.cache', 'huggingface')
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(HF_CACHE_DIR, 'datasets')

import torch
import torch.nn as nn
import torch.nn.functional as F
import datasets
import espnet
import espnetez as ez
import numpy as np
import yaml
from espnet2.bin.s2t_inference import Speech2Text

print("ESPnet version:", espnet.__version__)

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

        weights = F.softmax(logits, dim=-1)


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
    """Pre-Hook for ENCODER layers (Simple: only one input source)."""
    def __init__(self, gating_module: TokenLevelGating, layer_modules: List[MultiExpertLoRALinear]):
        self.gating_module = gating_module
        self.layer_modules = layer_modules

    def __call__(self, module, args):
        # Encoder args: (x, masks)
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


def load_adapter_checkpoint(path: str, prefix: str = "") -> Dict[str, torch.Tensor]:
    sd = torch.load(path, map_location='cpu')
    if prefix:
        filtered = {}
        for k, v in sd.items():
            if k.startswith(prefix):
                filtered[k[len(prefix):]] = v
        return filtered
    return sd

def replace_linear_with_multi_expert(
    module: nn.Module,
    target_modules: List[str],
    num_experts: int,
    rank: int,
    alpha: float,
    expert_params: List[dict]
) -> Dict[str, MultiExpertLoRALinear]:
    """Recursively replace Linear layers."""
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

def setup_moe_gating(
    model: nn.Module,
    num_experts: int,
    adapter_checkpoints: List[str],
    adapter_config: dict,
    encoder_hidden_dim: int,
    decoder_hidden_dim: int,
    target_lang: str = None
):
    target_modules = adapter_config['adapter_conf']['target_modules']
    rank = adapter_config['adapter_conf']['rank']
    alpha = adapter_config['adapter_conf']['alpha']

    enc_params = [load_adapter_checkpoint(p, "encoder.") for p in adapter_checkpoints]
    dec_params = [load_adapter_checkpoint(p, "decoder.") for p in adapter_checkpoints]

    enc_params.append({}) 
    dec_params.append({})

    print("Replacing Encoder Modules:")
    enc_replaced = replace_linear_with_multi_expert(
        model.encoder, target_modules, num_experts, rank, alpha, enc_params
    )
    
    # Organize Encoder Modules by Layer
    num_enc_layers = len(model.encoder.encoders)
    enc_layer_modules = [[] for _ in range(num_enc_layers)]
    for name, mod in enc_replaced.items():
        if "encoders." in name:
            idx = int(name.split("encoders.")[1].split('.')[0])
            enc_layer_modules[idx].append(mod)

    # Create Encoder Gating
    enc_gating_nets = nn.ModuleList([
        TokenLevelGating(encoder_hidden_dim, num_experts, target_lang = target_lang, mean_pooling=False) for _ in range(num_enc_layers)
    ])

    print("Replacing Decoder Modules:")
    dec_replaced = replace_linear_with_multi_expert(
        model.decoder, target_modules, num_experts, rank, alpha, dec_params
    )

    # Organize Decoder modules by layer and type
    num_dec_layers = len(model.decoder.decoders)
    dec_std_modules = [[] for _ in range(num_dec_layers)]
    dec_mem_modules = [[] for _ in range(num_dec_layers)]

    for name, mod in dec_replaced.items():
        if "decoders." in name:
            idx = int(name.split("decoders.")[1].split('.')[0])
            
            is_cross_kv = ("src_attn" in name) and (name.endswith("linear_k") or name.endswith("linear_v"))
            
            if is_cross_kv:
                dec_mem_modules[idx].append(mod)
            else:
                dec_std_modules[idx].append(mod)

    dec_gating_nets_x = nn.ModuleList([
        TokenLevelGating(decoder_hidden_dim, num_experts, target_lang = target_lang, mean_pooling=False) for _ in range(num_dec_layers)
    ])
    dec_gating_nets_mem = nn.ModuleList([
        TokenLevelGating(encoder_hidden_dim, num_experts, target_lang = target_lang, mean_pooling=False) for _ in range(num_dec_layers)
    ])

    return (
        enc_gating_nets, enc_layer_modules,
        dec_gating_nets_x, dec_gating_nets_mem, 
        dec_std_modules, dec_mem_modules
    )


def main(margs):
    os.makedirs(margs.exp_dir, exist_ok=True)
    if margs.stats_dir is None:
        margs.stats_dir = os.path.join(margs.exp_dir, "stats")
    
    # Config
    SOURCE_LANGS = ['cym', 'rus', 'ita', 'por', 'hun']
    ADAPTER_PATHS = [
        'exp/adapter_cym/adapters_only_cym.pt',
        'exp/adapter_rus/adapters_only_rus.pt',
        'exp/adapter_ita/adapters_only_ita.pt',
        'exp/adapter_por/adapters_only_por.pt',
        'exp/adapter_hun/adapters_only_hun.pt',
    ]

    # Dataset Setup
    try:
        fleurs_hf = datasets.load_dataset("google/fleurs", margs.fleurs_lang, trust_remote_code=True)
    except:
        fleurs_hf = datasets.load_dataset("google/fleurs", margs.fleurs_lang, trust_remote_code=True, download_mode="force_redownload")

    pretrained_model = Speech2Text.from_pretrained(
        margs.pretrained_model,
        lang_sym=f"<{margs.target_lang}>",
        beam_size=1,
        device='cuda'
    )
    
    pretrain_config = vars(pretrained_model.s2t_train_args)
    tokenizer = pretrained_model.tokenizer
    converter = pretrained_model.converter

    def tokenize(text):
        return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

    data_info = {
        "speech": lambda d: d['audio']['array'].astype(np.float32),
        "text": lambda d: tokenize(f"<{margs.target_lang}><asr><notimestamps> {d['transcription']}"),
        "text_ctc": lambda d: tokenize(d['transcription']),
        "text_prev": lambda d: tokenize("<na>"),
    }
    
    train_dataset = ez.dataset.ESPnetEZDataset(fleurs_hf['train'], data_info=data_info)
    valid_dataset = ez.dataset.ESPnetEZDataset(fleurs_hf['validation'], data_info=data_info)

    with open(margs.finetune_config, 'r') as f:
        adapter_config = yaml.safe_load(f)

    def build_model_fn(args):
        model = pretrained_model.s2t_model
        enc_dim = model.encoder.encoders[0].size
        dec_dim = model.decoder.decoders[0].size
        
        print(f"Dimensions - Enc: {enc_dim}, Dec: {dec_dim}")

        (enc_gates, enc_mods, 
         dec_gates_x, dec_gates_mem, 
         dec_std_mods, dec_mem_mods) = setup_moe_gating(
            model, len(SOURCE_LANGS) + 1, ADAPTER_PATHS, adapter_config, enc_dim, dec_dim,
        )

        # 2. Register Parameters
        device = next(model.parameters()).device
        
        if not hasattr(model, 'moe_components'):
            model.moe_components = nn.ModuleDict({
                'enc_gates': enc_gates.to(device),
                'dec_gates_x': dec_gates_x.to(device),
                'dec_gates_mem': dec_gates_mem.to(device),
            })

        # Clear existing hooks
        for l in model.encoder.encoders: l._forward_pre_hooks.clear()
        for l in model.decoder.decoders: l._forward_pre_hooks.clear()

        # Encoder Hooks
        for i, layer in enumerate(model.encoder.encoders):
            if enc_mods[i]:
                hook = EncoderMoEPreHook(model.moe_components['enc_gates'][i], enc_mods[i])
                layer.register_forward_pre_hook(hook)

        # Decoder Hooks (Dual Gating)
        for i, layer in enumerate(model.decoder.decoders):
            if dec_std_mods[i] or dec_mem_mods[i]:
                hook = DecoderMoEPreHook(
                    model.moe_components['dec_gates_x'][i],
                    model.moe_components['dec_gates_mem'][i],
                    dec_std_mods[i],
                    dec_mem_mods[i]
                )
                layer.register_forward_pre_hook(hook)

        # 4. Freeze & Unfreeze
        model.train()
        for p in model.parameters(): 
            p.requires_grad = False
        
        # Unfreeze Gating
        trainable = 0
        for p in model.moe_components.parameters():
            p.requires_grad = True
            trainable += p.numel()
            
        print(f"Trainable params: {trainable}")
        return model

    finetune_config = ez.config.update_finetune_config('s2t', pretrain_config, margs.finetune_config)
    
    trainer = ez.Trainer(
        task='s2t',
        train_config=finetune_config,
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        build_model_fn=build_model_fn,
        data_info=data_info,
        output_dir=margs.exp_dir,
        stats_dir=margs.stats_dir,
        ngpu=1
    )

    trainer.collect_stats()
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fleurs_lang", required=True)
    parser.add_argument("--target_lang", required=True)
    parser.add_argument("--exp_dir", required=True)
    parser.add_argument("--stats_dir", default=None)
    parser.add_argument("--finetune_config", default="config/finetune_adapter.yaml")
    parser.add_argument("--pretrained_model", default="espnet/owsm_v4_small_370M")
    main(parser.parse_args())