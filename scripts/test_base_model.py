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
from espnet2.bin.s2t_inference import Speech2Text


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


def test_base_model(fleurs_lang, owsm_lang, pretrained_model="espnet/owsm_v4_base_102M", num_samples=None, device='cuda'):
    """Test the base pretrained model on a specific language's test set"""

    print(f"Loading base model: {pretrained_model}")
    print(f"Testing on: {fleurs_lang} (OWSM language: {owsm_lang})")

    model = Speech2Text.from_pretrained(
        pretrained_model,
        lang_sym=f"<{owsm_lang}>",
        beam_size=1,
        device=device
    )

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

    print(f"Testing on {len(test_data)} samples...\n")

    references = []
    hypotheses = []

    for i, sample in enumerate(test_data):

        audio = sample['audio']['array'].astype(np.float32)
        reference = sample['transcription']

        with torch.no_grad():
            results = model(audio)

        hypothesis = results[0][0] if results and len(results) > 0 and len(results[0]) > 0 else ""

        # Strip prompt tokens from hypothesis
        if hypothesis.startswith("<"):
            # Find the last '>' and take everything after it
            last_bracket = hypothesis.rfind('>')
            if last_bracket != -1:
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

    wer, cer, references, hypotheses = test_base_model(
        args.fleurs_lang,
        args.owsm_lang,
        pretrained_model=args.pretrained_model,
        num_samples=args.num_samples,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )


    if args.output_file:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(f"Base Model Evaluation Results\n")
            f.write(f"{'='*80}\n")
            f.write(f"Model: {args.pretrained_model}\n")
            f.write(f"Language: {args.fleurs_lang} (OWSM: {args.owsm_lang})\n")
            f.write(f"Samples: {len(references)}\n")
            f.write(f"{'='*80}\n\n")
            f.write(f"WER: {wer:.4f} ({wer*100:.2f}%)\n")
            f.write(f"CER: {cer:.4f} ({cer*100:.2f}%)\n\n")
            f.write(f"{'='*80}\n")
            f.write(f"Detailed Results\n")
            f.write(f"{'='*80}\n\n")

            for i, (ref, hyp) in enumerate(zip(references[:5], hypotheses[:5])):
                f.write(f"Sample {i+1}:\n")
                f.write(f"  REF: {ref}\n")
                f.write(f"  HYP: {hyp}\n\n")

        print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate base (non-finetuned) OWSM model on a FLEURS language"
    )

    # Language info
    parser.add_argument(
        "--fleurs_lang",
        type=str,
        required=True,
        help="FLEURS language code to test on (e.g., pt_br, es_419, it_it)",
    )

    parser.add_argument(
        "--owsm_lang",
        type=str,
        required=True,
        help="OWSM language code (e.g., por, spa, ita)",
    )

    # Model
    parser.add_argument(
        "--pretrained_model",
        type=str,
        default="espnet/owsm_v4_base_102M",
        help="Pretrained OWSM model to evaluate (default: espnet/owsm_v4_base_102M)",
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
