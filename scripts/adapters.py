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
import espnetez as ez # ESPnet wrapper that simplifies integration. If you get an error when executing this cell, click Runtime -> Restart Session, and rerun from the beginning
import numpy as np
import librosa
from espnet2.bin.s2t_inference import Speech2Text # Core ESPnet module for pre-trained models
from espnet2.layers.create_adapter import create_adapter

print("Successfully imported espnet ez")
print("ESPnet version: ", espnet.__version__)


def main(arguments):

    os.makedirs(arguments.exp_dir, exist_ok=True)
    if arguments.stats_dir is None:
        arguments.stats_dir = os.path.join(arguments.exp_dir, "stats")
    os.makedirs(arguments.stats_dir, exist_ok=True)


    try:
        fleurs_hf = datasets.load_dataset(
            "google/fleurs",
            arguments.fleurs_lang,
            trust_remote_code=True 
        )
    except ValueError as e:
        fleurs_hf = datasets.load_dataset(
            "google/fleurs",
            arguments.fleurs_lang,
            trust_remote_code=True,
            download_mode="force_redownload"  
        )



    pretrained_model = Speech2Text.from_pretrained(
        arguments.pretrained_model,
        lang_sym=f"<{arguments.owsm_lang}>",
        beam_size=1,
        device='cuda'
    )

    # Save checkpoint for comparison between original model vs finetuned model
    # torch.save(pretrained_model.s2t_model.state_dict(), 'original.pth')

    # Extracting configs and tokenizers from pretrained model
    pretrain_config = vars(pretrained_model.s2t_train_args)
    tokenizer = pretrained_model.tokenizer
    converter = pretrained_model.converter


    """
    pretrained_model -> the pre-trained model we downloaded earlier
    tokenizer -> Tokenizes raw text into subwords
    converter -> Converts subwords into integer IDs for model input
    """

    def tokenize(text):
        return np.array(converter.tokens2ids(tokenizer.text2tokens(text)))

    data_info = {
        "speech"    : lambda d: d['audio']['array'].astype(np.float32),
        "text"      : lambda d: tokenize(
                        f"<{arguments.owsm_lang}><asr><notimestamps> {d['transcription']}"
                    ),
        "text_ctc"  : lambda d: tokenize(d['transcription']),
        "text_prev" : lambda d: tokenize("<na>"),
    }

    # We keep "text_raw" for raw untokenized text as the reference.
    # We will not give this valie to the model.
    test_data_info = {
        "speech"    : lambda d: d['audio']['array'].astype(np.float32),
        "text"      : lambda d: tokenize(
                        f"<{arguments.owsm_lang}><asr><notimestamps> {d['transcription']}"
                    ),
        "text_ctc"  : lambda d: tokenize(d['transcription']),
        "text_prev" : lambda d: tokenize("<na>"),
        "text_raw"  : lambda d: d['transcription'],
    }

    # Then wrap any dataset class with ESPnetEZDataset class.
    # Each data will be mapped to ESPnet's dataset on-the-fly.
    train_dataset = ez.dataset.ESPnetEZDataset(fleurs_hf['train'], data_info=data_info)
    valid_dataset = ez.dataset.ESPnetEZDataset(fleurs_hf['validation'], data_info=data_info)
    test_dataset  = ez.dataset.ESPnetEZDataset(fleurs_hf['test'], data_info=test_data_info)

    print("Train dataset first item:", train_dataset[0])
    print("Validation dataset first item:", valid_dataset[0])

    # define model loading function
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def build_model_fn(args):
        model = pretrained_model.s2t_model

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
        train_dataset=train_dataset,
        valid_dataset=valid_dataset,
        build_model_fn=build_model_fn, # provide the pre-trained model
        data_info=data_info,
        output_dir=arguments.exp_dir,
        stats_dir=arguments.stats_dir,
        ngpu=1
    )

    trainer.collect_stats()

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--fleurs_lang",
        type=str,
        required=True,
        help="FLEURS language code, e.g. en_us",
    )

    parser.add_argument(
        "--owsm_lang",
        type=str,
        required=True,
        help="ISO3 language code for OWSM prompts, e.g. eng",
    )

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
        default="config/finetune_adapter.yaml",
        help="YAML with adapter + training hyperparameters",
    )

    # Which pretrained model to adapt
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="espnet/owsm_v4_small_370M",
        help="Pretrained OWSM v4 model ID",
    )

    arguments = parser.parse_args()

    main(arguments)