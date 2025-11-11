#!/usr/bin/env python3

import os
import argparse

# Set cache directories BEFORE importing datasets/HF libraries
SCRIPT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HF_CACHE_DIR = os.path.join(SCRIPT_DIR, '.cache', 'huggingface')
os.environ['HF_HOME'] = HF_CACHE_DIR
os.environ['HF_DATASETS_CACHE'] = os.path.join(HF_CACHE_DIR, 'datasets')

import torch
import datasets
import espnet
import espnetez as ez
import numpy as np
import librosa
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.layers.create_adapter import create_adapter

print("Successfully imported espnet ez")
print("ESPnet version: ", espnet.__version__)


def main(arguments):

    os.makedirs(arguments.exp_dir, exist_ok=True)
    if arguments.stats_dir is None:
        arguments.stats_dir = os.path.join(arguments.exp_dir, "stats")
    os.makedirs(arguments.stats_dir, exist_ok=True)

    languages = [
        ("hu_hu", "hun"),
        ("ru_ru", "rus"),
        ("cy_gb", "cym"),
        ("it_it", "ita"),
        ("pt_br", "por"),
    ]


    pretrained_model = Speech2Text.from_pretrained(
        arguments.pretrained_model,
        lang_sym=f"<{languages[0][1]}>",  # Use first language symbol?
        beam_size=1,
        device='cuda'
    )

    # Extracting configs and tokenizers from pretrained model
    pretrain_config = vars(pretrained_model.s2t_train_args)
    tokenizer = pretrained_model.tokenizer
    converter = pretrained_model.converter

    def tokenize(text):
        return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

    merged_train = []
    merged_valid = []

    for fleurs_lang, owsm_lang in languages:
        fleurs_hf = datasets.load_dataset(
            "google/fleurs",
            fleurs_lang,
            trust_remote_code=True
        )

        # Add lang tag to each example
        for ex in fleurs_hf["train"]:
            merged_train.append({
                "audio": ex["audio"],
                "transcription": ex["transcription"],
                "lang": owsm_lang,
            })

        for ex in fleurs_hf["validation"]:
            merged_valid.append({
                "audio": ex["audio"],
                "transcription": ex["transcription"],
                "lang": owsm_lang,
            })

    data_info = {
        "speech": lambda d: d["audio"]["array"].astype(np.float32),
        "text": lambda d: tokenize(
            f"<{d['lang']}><asr><notimestamps> {d['transcription']}"
        ),
        "text_ctc": lambda d: tokenize(d["transcription"]),
        "text_prev": lambda d: tokenize("<na>"),
    }

    train_ds = ez.dataset.ESPnetEZDataset(merged_train, data_info=data_info)
    valid_ds = ez.dataset.ESPnetEZDataset(merged_valid, data_info=data_info)

    # define model loading function
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def build_model_fn(args):
        model = pretrained_model.s2t_model

        if getattr(args, "use_adapter", False):
            create_adapter(model, args.adapter, args.adapter_conf)

        model.train()

        for p in model.parameters():
            p.requires_grad = False

        trainable_count = 0
        lora_modules = []
        for n, p in model.named_parameters():
            if ("lora" in n or "adapter" in n):
                p.requires_grad = True
                trainable_count += p.numel()

        print(f'Trainable parameters: {trainable_count:,}')

        return model

    finetune_config = ez.config.update_finetune_config(
        's2t',
        pretrain_config,
        arguments.finetune_config
    )

    trainer = ez.Trainer(
        task='s2t',
        train_config=finetune_config,
        train_dataset=train_ds,  
        valid_dataset=valid_ds,  
        build_model_fn=build_model_fn,
        data_info=None,  # Already applied
        output_dir=arguments.exp_dir,
        stats_dir=arguments.stats_dir,
        ngpu=1
    )

    trainer.collect_stats()

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Paths
    parser.add_argument(
        "--exp_dir",
        type=str,
        required=True,
        help="Directory to save checkpoints/logs for this run",
    )

    parser.add_argument(
        "--stats_dir",
        type=str,
        default=None,
        help="Directory for stats (default: <exp_dir>/stats)",
    )

    parser.add_argument(
        "--finetune_config",
        type=str,
        default="config/finetune_adapter_shared.yaml",
        help="YAML with adapter + training hyperparameters",
    )

    # Which pretrained model to adapt
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="espnet/owsm_v4_base_102M",
        help="Pretrained OWSM CTC v4 model ID",
    )

    arguments = parser.parse_args()

    main(arguments)
