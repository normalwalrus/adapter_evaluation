from peft import PeftModel, PeftConfig
import librosa
from datasets import load_dataset
from transformers import (WhisperForConditionalGeneration, 
                          Seq2SeqTrainer, 
                          BitsAndBytesConfig, 
                          WhisperTokenizer, 
                          WhisperProcessor,
                           WhisperFeatureExtractor,
                          AutomaticSpeechRecognitionPipeline)
from utils.data_collator import DataCollatorSpeechSeq2SeqWithPadding
from transformers import WhisperProcessor
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import numpy as np
import gc
import os
import evaluate

################################################## EDIT HERE #################################################################################
checkpoints_dirpath = "/adpt-test/checkpoints/whisper_v3_GV_LoRA_r16_15epochs_full_0.00005lrhi_run/" # NOTE: This is the path to the folder containing all the adapter checkpoints
base_model_path = '/adpt-test/pretrained_models/whisper-large-v3-multilingual-50h/checkpoint-13616'
path_to_test_data = '/adpt-test/data/GV_hindi/hi/'
test_manifest = 'eval_manifest.json'

lang = 'Hindi'
task = 'transcribe'

eval_metric = 'wer'
output_filepath = f"outputs/whisper_GV_LoRA_15epochs_{lang}_0.00001lr_32r_run.txt"

batch_size = 8
###############################################################################################################################################

SAMPLING_RATE = 16000
metric = evaluate.load(eval_metric)

def read_audio(batch, path_to_data_folder):

    filepath = path_to_data_folder + batch['audio_filepath']
    audio, _ = librosa.load(filepath, sr=16000, mono=True)
    assert len(audio.shape) == 1
    batch['input_values'] = audio
    batch['input_length'] = len(batch['input_values'])

    return batch

def prepare_dataset(batch, feature_extractor, tokenizer):
    # compute log-Mel input features from input audio array
    batch["input_features"] = feature_extractor(batch["input_values"], sampling_rate=SAMPLING_RATE).input_features[0]

    # encode target text to label ids
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch

# Initialise the model
model = WhisperForConditionalGeneration.from_pretrained(
   base_model_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
)
processor = WhisperProcessor.from_pretrained(base_model_path, language=lang, task=task)
data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

# Data prepping
path_to_test_manifest = os.path.join(path_to_test_data, test_manifest)

test_data = load_dataset("json", data_files=path_to_test_manifest, split= 'train')
columns_test = test_data.column_names
columns_test.remove('audio_filepath')
columns_test.remove('text')

test_data = test_data.remove_columns(columns_test)
updated_test_data = test_data.map(read_audio, fn_kwargs={"path_to_data_folder": path_to_test_data})

feature_extractor = WhisperFeatureExtractor.from_pretrained(base_model_path, language=lang, task=task)
tokenizer = WhisperTokenizer.from_pretrained(base_model_path, language=lang, task=task)

updated_test_data = updated_test_data.map(prepare_dataset, fn_kwargs={"feature_extractor": feature_extractor, "tokenizer":tokenizer})

columns = updated_test_data.column_names
columns.remove('input_features')
columns.remove('labels')

updated_test_data = updated_test_data.remove_columns(columns)
eval_dataloader = DataLoader(updated_test_data, batch_size=batch_size, collate_fn=data_collator)

# Evaluation loop
checkpoints = [f for f in os.listdir(checkpoints_dirpath) if os.path.isdir(os.path.join(checkpoints_dirpath, f))]
score_keeping = {}

for checkpoint in checkpoints:
    print(f"For {checkpoint} : ") 
    peft_model_id = os.path.join(checkpoints_dirpath, checkpoint)
    
    model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path, quantization_config=BitsAndBytesConfig(load_in_8bit=True), device_map="auto"
    )
    model = PeftModel.from_pretrained(model, peft_model_id)

    processor = WhisperProcessor.from_pretrained(base_model_path, language=lang, task=task)
    model.eval()
    for step, batch in enumerate(tqdm(eval_dataloader)):
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
                decoded_preds = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
                
                metric.add_batch(
                    predictions=decoded_preds,
                    references=decoded_labels,
                )
        del generated_tokens, labels, batch
        gc.collect()

    cer = 100 * metric.compute()
    print(f"For {checkpoint} : WER = {cer}")
    score_keeping[checkpoint] = cer
    
    with torch.no_grad():
        torch.cuda.empty_cache()

# Save the evaluation results in a txt file
with open(output_filepath, "w") as file:
    for key, value in score_keeping.items():
        file.write(f"{key}: {value}\n")