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
import numpy as np
import jiwer
import re
import yaml
from espnet2.bin.s2t_inference import Speech2Text
from espnet2.layers.create_adapter import create_adapter


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
    """Normalize text: transliterate (if Serbian), lowercase, remove punctuation, strip whitespace"""
    if owsm_lang == 'srp':
        text = cyrillic_to_latin_serbian(text)

    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def calculate_wer(references, hypotheses):
    """Calculate Word Error Rate - assumes inputs are already normalized"""
    wer = jiwer.wer(references, hypotheses)
    return wer


def calculate_cer(references, hypotheses):
    """Calculate Character Error Rate - assumes inputs are already normalized"""
    cer = jiwer.cer(references, hypotheses)
    return cer


def load_finetuned_model(checkpoint_path, test_owsm_lang, adapter_config_path, device='cuda'):
    """Load a finetuned model from checkpoint"""
    # Load the pretrained model with the test language prompt
    print(f"\nLoading base model with language token: <{test_owsm_lang}>")
    model = Speech2Text.from_pretrained(
        "espnet/owsm_v4_small_370M",
        lang_sym=f"<{test_owsm_lang}>",
        beam_size=5,
        device=device
    )


    # Load adapter configuration and create adapters
    with open(adapter_config_path, 'r') as f:
        config = yaml.safe_load(f)

    create_adapter(
        model.s2t_model,
        config['adapter'],
        config['adapter_conf']
    )

    
    # Load state dict and capture what was loaded
    checkpoint = torch.load(checkpoint_path, map_location=device)
    missing_keys, unexpected_keys = model.s2t_model.load_state_dict(checkpoint, strict=False)
    print(f"Loaded checkpoint from: {checkpoint_path}")

    adapter_params_loaded = 0
    for name, param in model.s2t_model.named_parameters():
        if ('lora' in name or 'adapter' in name) and name in checkpoint:
            adapter_params_loaded += param.numel()
    print(f"Adapter parameters loaded from checkpoint: {adapter_params_loaded}")

    model.s2t_model.eval()

    return model


def test_model_on_language(model, fleurs_lang, owsm_lang, num_samples=None):
    """Test a model on a specific language's test set"""

    try:
        fleurs_hf = datasets.load_dataset(
            "google/fleurs",
            fleurs_lang,
            trust_remote_code=True
        )
    except ValueError:
        fleurs_hf = datasets.load_dataset(
            "google/fleurs",
            fleurs_lang,
            trust_remote_code=True,
            download_mode="force_redownload"
        )

    test_data = fleurs_hf['test']

    # Limit samples if specified
    if num_samples is not None:
        test_data = test_data.select(range(min(num_samples, len(test_data))))

    references = []
    hypotheses = []

    print(f"\nTesting on {fleurs_lang} ({len(test_data)} samples)...")

    for i, sample in enumerate(test_data):
        
        audio = sample['audio']['array'].astype(np.float32)
        reference = sample['transcription']

        with torch.no_grad():
            results = model(audio)


        hypothesis = results[0][0] if (results and len(results) > 0 and len(results[0]) > 0) else ""

        # Strip prompt tokens from hypothesis
        if hypothesis.startswith("<"):
            # Find the last '>' and take everything after it
            last_bracket = hypothesis.rfind('>')
            if last_bracket != -1:
                hypothesis_before_strip = hypothesis
                hypothesis = hypothesis[last_bracket + 1:].strip()

        # Normalize both reference and hypothesis
        reference_normalized = normalize_text(reference, owsm_lang=owsm_lang)
        hypothesis_normalized = normalize_text(hypothesis, owsm_lang=owsm_lang)

        references.append(reference_normalized)
        hypotheses.append(hypothesis_normalized)


    wer = calculate_wer(references, hypotheses)
    cer = calculate_cer(references, hypotheses)

    return wer, cer, references, hypotheses


def main(args):

    model = load_finetuned_model(
        args.checkpoint,
        args.owsm_lang,  # Use test language prompt
        args.adapter_config,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Test on the target language
    wer, cer, references, hypotheses = test_model_on_language(
        model,
        args.fleurs_lang,
        args.owsm_lang,
        num_samples=args.num_samples
    )

    # Save results to file if requested
    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Checkpoint Evaluation Results\n")
            f.write(f"Model checkpoint: {args.checkpoint}\n")
            f.write(f"Testing on: {args.fleurs_lang} (OWSM: {args.owsm_lang})\n")
            f.write(f"Samples: {len(references)}\n")
            f.write(f"WER: {wer:.4f} ({wer*100:.2f}%)\n")
            f.write(f"CER: {cer:.4f} ({cer*100:.2f}%)\n\n")

            f.write(f"Examples:\n\n")

            for i, (ref, hyp) in enumerate(zip(references[:5], hypotheses[:5])):
                f.write(f"Sample {i+1}:\n")
                f.write(f"  REF: {ref}\n")
                f.write(f"  HYP: {hyp}\n\n")

        print(f"\nResults saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate a model checkpoint on a target language"
    )

    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to finetuned model checkpoint (e.g., exp/adapter_por/valid.total_count.best.pth)",
    )

    # Adapter configuration
    parser.add_argument(
        "--adapter_config",
        type=str,
        default="config/finetune_adapter.yaml",
        help="Path to adapter config YAML (default: config/finetune_adapter.yaml)",
    )

    # Test language info
    parser.add_argument(
        "--fleurs_lang",
        type=str,
        required=True,
        help="FLEURS language code to test on (e.g., pt_br)",
    )

    parser.add_argument(
        "--owsm_lang",
        type=str,
        required=True,
        help="OWSM language code to test with (e.g., por)",
    )

    # Optional parameters
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of test samples to use (default: all)",
    )

    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Path to save detailed results (default: print to console only)",
    )

    args = parser.parse_args()
    main(args)
