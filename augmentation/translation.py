import os
import json
import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "")


def translate_to_zh(text, model, tokenizer):
    prefix = 'translate to zh: '
    source_text = prefix + text
    input_ids = tokenizer(source_text, return_tensors="pt")
    generated_tokens = model.generate(**input_ids.to(device))
    result = tokenizer.batch_decode(generated_tokens)
    return result[0].replace("<pad>", "").replace("</s>", "")


def translate_to_en(text, model, tokenizer):
    prefix = 'translate to en: '
    source_text = prefix + text
    input_ids = tokenizer(source_text, return_tensors="pt")
    generated_tokens = model.generate(**input_ids.to(device))
    result = tokenizer.batch_decode(generated_tokens)
    return result[0].replace("<pad>", "").replace("</s>", "")


def translation(origin_dataset, language, percentage=0.1):
    model = T5ForConditionalGeneration.from_pretrained('utrobinmv/t5_translate_en_ru_zh_large_1024').to(device)
    tokenizer = T5Tokenizer.from_pretrained('utrobinmv/t5_translate_en_ru_zh_large_1024')

    augmented_dataset = []
    dataset_len = int(len(origin_dataset) * percentage)

    for idx, data in enumerate(origin_dataset[:dataset_len]):
        augmented_dataset.append(data)

    offset = len(augmented_dataset)

    for item in tqdm(origin_dataset[:dataset_len]):
        idx = item["id"] + offset
        target_text = item['target_text']

        if language == "zh":
            translated_text = translate_to_en(target_text, model, tokenizer)
            translated_text = translate_to_zh(translated_text, model, tokenizer)
        else:
            translated_text = translate_to_zh(target_text, model, tokenizer)
            translated_text = translate_to_en(translated_text, model, tokenizer)

        new_case = {"id": idx, "source_text": translated_text.strip(), "target_text": target_text.strip()}
        augmented_dataset.append(new_case)

    return augmented_dataset
