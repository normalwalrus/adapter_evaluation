"""
Filename: lora_testing.py
Author: Digital Hub
Date: 2025-06-11
Version: 0.1.0
Description: 
    provides lora checkpoint evaluation for whisper
"""

import gc
import json
import logging
import os

import jiwer
import librosa
import numpy as np
import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
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
from utils.lang_utils import filter_string

logging.basicConfig()
logging.getLogger().setLevel(logging.INFO)

with open("/adpt-test/config.yml", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PATH_TO_CHECKPOINTS = config["eval"][
    "path_to_checkpoints"
]  # NOTE: This is the path to the folder containing all the adapter checkpoints
PATH_TO_BASE_MODEL = config["eval"]["path_to_base_model"]
PATH_TO_TEST_DATA = config["eval"]["path_to_test_data"]
NAME_OF_TEST_MANIFEST = config["eval"]["name_of_test_manifest"]
NAME_OF_OUTPUT_FILE = config["eval"]["name_of_output_file"]

LANG = config["eval"]["lang"]
TASK = config["eval"]["task"]

BATCH_SIZE = config["eval"]["batch_size"]
CHECKPOINT_CHOSEN = config["eval"]["checkpoint_chosen"]

SAMPLING_RATE = 16000
# OUTPUT_FILEPATH = os.path.join("outputs", NAME_OF_OUTPUT_FILE)

print(config)

def read_audio(batch, path_to_data_folder: str):
    ''' Takes in audio filepath in hugginface dataset to get signal '''
    filepath = path_to_data_folder + batch["audio_filepath"]
    audio, _ = librosa.load(filepath, sr=16000, mono=True)
    assert len(audio.shape) == 1
    batch["input_values"] = audio
    batch["input_length"] = len(batch["input_values"])

    return batch


def prepare_dataset(batch, feature_extractor, tokenizer):
    ''' Converts audio signal into whisper features for evaluation '''
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(
        batch["input_values"], sampling_rate=SAMPLING_RATE
    ).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def lora_eval():
    ''' Main lora evluation loop '''
    # Initialise the model
    model = WhisperForConditionalGeneration.from_pretrained(
        PATH_TO_BASE_MODEL,
        quantization_config=BitsAndBytesConfig(load_in_8bit=True),
        device_map="auto",
    )
    processor = WhisperProcessor.from_pretrained(
        PATH_TO_BASE_MODEL, language=LANG, task=TASK
    )
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    # Data prepping
    path_to_test_manifest = os.path.join(PATH_TO_TEST_DATA, NAME_OF_TEST_MANIFEST)

    test_data = load_dataset("json", data_files=path_to_test_manifest, split="train")
    columns_test = test_data.column_names
    columns_test.remove("audio_filepath")
    columns_test.remove("text")

    test_data = test_data.remove_columns(columns_test)
    updated_test_data = test_data.map(
        read_audio, fn_kwargs={"path_to_data_folder": PATH_TO_TEST_DATA}
    )

    feature_extractor = WhisperFeatureExtractor.from_pretrained(
        PATH_TO_BASE_MODEL, language=LANG, task=TASK
    )
    tokenizer = WhisperTokenizer.from_pretrained(
        PATH_TO_BASE_MODEL, language=LANG, task=TASK
    )

    updated_test_data = updated_test_data.map(
        prepare_dataset,
        fn_kwargs={"feature_extractor": feature_extractor, "tokenizer": tokenizer},
    )

    eval_dataloader = DataLoader(
        updated_test_data, batch_size=BATCH_SIZE, collate_fn=data_collator
    )

    # Evaluation loop
    checkpoints = [
        f
        for f in os.listdir(PATH_TO_CHECKPOINTS)
        if os.path.isdir(os.path.join(PATH_TO_CHECKPOINTS, f))
    ]

    for checkpoint in checkpoints:

        if CHECKPOINT_CHOSEN:
            if checkpoint != CHECKPOINT_CHOSEN:
                continue

        print(f"For {checkpoint} : ")
        peft_model_id = os.path.join(PATH_TO_CHECKPOINTS, checkpoint)
        
        model = WhisperForConditionalGeneration.from_pretrained(
            PATH_TO_BASE_MODEL,
            quantization_config=BitsAndBytesConfig(load_in_8bit=True),
            device_map="auto",
        )
        model = PeftModel.from_pretrained(model, peft_model_id)

        output_filepath = os.path.join("outputs", checkpoint+'.json')
        
        processor = WhisperProcessor.from_pretrained(
            PATH_TO_BASE_MODEL, language=LANG, task=TASK
        )
        model.eval()
        references_proc = []
        predictions_proc = []
        for _, batch in enumerate(tqdm(eval_dataloader)):
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

                    pred_str_processed = [
                        filter_string(x, lang=LANG) for x in decoded_preds
                    ]
                    label_str_processed = [
                        filter_string(x, lang=LANG) for x in decoded_labels
                    ]
                    
                    for pred, ref, pred_proc, ref_proc, path in zip(
                        decoded_preds,
                        decoded_labels,
                        pred_str_processed,
                        label_str_processed,
                        batch["audio_filepath"],
                    ):
                        with open(output_filepath, "a", encoding="utf-8") as file:
                            file.write(
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
        cer_overall = jiwer.cer(reference=references_proc, hypothesis=predictions_proc)

        logging.info("[Overall WER]: %s", round(wer_overall, 3))
        logging.info("[Overall CER]: %s", round(cer_overall, 3))

        with torch.no_grad():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    lora_eval()
