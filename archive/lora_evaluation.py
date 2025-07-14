"""
Script to run evaluation of a single adapter checkpoint and write results to manifest
file
"""

import gc
import json
import logging
import os
import re

import jiwer
import librosa
import numpy as np
import torch
from datasets import load_dataset
from peft import PeftConfig, PeftModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

################################################## EDIT HERE #################################################################################
checkpoint_filepath = (
    "/adpt-test/models/whisper/whisper_en_port_adapters/checkpoint-22725"
)
base_model_path = "/adpt-test/models/whisper/whisper_en_port_set_1_2/checkpoint-17100"
path_to_test_data = "/adpt-test/data/PORT/mms_set_3/test_split"
test_manifest = "test_manifest_smol.json"

lang = "English"
task = "transcribe"

output_filepath = f"outputs/whisper-adapter-pred.json"
os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
batch_size = 8
###############################################################################################################################################

SAMPLING_RATE = 16000


def read_audio(batch, path_to_data_folder):

    filepath = os.path.join(path_to_data_folder, batch["audio_filepath"])
    audio, _ = librosa.load(filepath, sr=16000, mono=True)
    assert len(audio.shape) == 1
    batch["input_values"] = audio
    batch["input_length"] = len(batch["input_values"])

    return batch


def prepare_dataset(batch, feature_extractor, tokenizer):
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        batch["input_values"], sampling_rate=SAMPLING_RATE
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def filter_string(input_string: str) -> str:
    """
    Filter the string such that comparison for WER and evaluation metrics are accurate.
    Current filters:
        - Lowercase all characters
        - Only keep alphabetic characters and whitespaces

    Parameters:
    input_string (str): The string to be filtered.

    Returns:
    String: The filtered string.
    """
    # Convert both texts to lowercase
    input_string = input_string.lower()
    # Use regular expression to find all alphabetic characters and whitespaces
    filtered_string = re.sub(r"[^a-zA-Z\s]", "", input_string)
    return filtered_string


def lora_eval():
    # Initialise the model
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    processor = WhisperProcessor.from_pretrained(
        base_model_path, language=lang, task=task
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Data prepping
    path_to_test_manifest = os.path.join(path_to_test_data, test_manifest)

    test_data = load_dataset("json", data_files=path_to_test_manifest, split="train")

    updated_test_data = test_data.map(
        read_audio, fn_kwargs={"path_to_data_folder": path_to_test_data}
    )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        base_model_path, language=lang, task=task
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        base_model_path, language=lang, task=task
    )

    updated_test_data = updated_test_data.map(
        prepare_dataset,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
    )

    eval_dataloader = DataLoader(
        updated_test_data, batch_size=batch_size, collate_fn=data_collator
    )

    peft_model_id = checkpoint_filepath

    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path,
        # quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    model = PeftModel.from_pretrained(model, peft_model_id)

    processor = WhisperProcessor.from_pretrained(
        base_model_path, language=lang, task=task
    )
    model.eval()
    references_proc = []
    predictions_proc = []
    for batch in tqdm(eval_dataloader):
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                generated_tokens = (
                    model.generate(
                        input_features=batch["input_features"].to("cuda"),
                        decoder_input_ids=batch["labels"][:, :4].to("cuda"),
                        max_new_tokens=255,
                    )
                    .cpu()
                    .numpy()
                )
                labels = batch["labels"].cpu().numpy()
                labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
                decoded_preds = tokenizer.batch_decode(
                    generated_tokens, skip_special_tokens=True
                )
                decoded_labels = tokenizer.batch_decode(
                    labels, skip_special_tokens=True
                )

                pred_str_processed = [filter_string(x) for x in decoded_preds]
                label_str_processed = [filter_string(x) for x in decoded_labels]
                for pred, ref, pred_proc, ref_proc, path in zip(
                    decoded_preds,
                    decoded_labels,
                    pred_str_processed,
                    label_str_processed,
                    batch["audio_filepath"],
                ):
                    with open(output_filepath, "a", encoding="utf-8") as f:
                        f.write(
                            json.dumps(
                                    {
                                        "audio_filepath": path,
                                        "prediction": pred,
                                        "reference": ref,
                                        "wer": jiwer.wer(
                                            reference=ref_proc,
                                            hypothesis=pred_proc,
                                        ),
                                        "cer": jiwer.cer(
                                            reference=ref_proc,
                                            hypothesis=pred_proc,
                                        ),
                                    },
                                    ensure_ascii=False,
                                )
                            + "\n"
                        )
                    references_proc.append(ref_proc)
                    predictions_proc.append(pred_proc)

        del generated_tokens, labels, batch
        gc.collect()
    wer_overall = jiwer.wer(reference=references_proc, hypothesis=predictions_proc)
    logging.info("[Overall WER]: %s", round(wer_overall, 3))

    with torch.no_grad():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    lora_eval()
