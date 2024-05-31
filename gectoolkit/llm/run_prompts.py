
import json
import os
import random
import torch
import datetime
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM

# Prompt
ICL_PROMPT_FOR_ENGLISH_GEC = """
Please correct all the grammatical errors and spelling mistakes in the given source sentences indicated by <input> ERROR </input> tag.
Before identifying and correcting any errors while keeping the original sentence structure unchanged as much as possible, you need to comprehend the sentence as a whole.
Afterward, return the corrected sentence directly without any explanations.
Remember to format your output results with tag <output> Your Corrected Version </output>.

Here are some examples for you: {}

Please understand these examples and correct the following sentence.

<input> {} </input>: 
"""

PROMPT_FOR_ENGLISH_GEC = """
Please correct all the grammatical errors and spelling mistakes in the given source sentences indicated by <input> ERROR </input> tag.
Before identifying and correcting any errors while keeping the original sentence structure unchanged as much as possible, you need to comprehend the sentence as a whole.
Afterward, return the corrected sentence directly without any explanations.
Remember to format your output results with tag <output> Your Corrected Version </output>.

Please correct the following sentence.

<input> {} </input>: 
"""

ICL_PROMPT_FOR_CHINESE_GEC = """
请纠正使用<input> 错误语句 </input>标签指出的句子中的所有语法错误和错别字。如果没有句子中没有错误，则直接返回原始句子。
请先将句子作为一个整体来理解，再进行纠错。纠错过程中，请尽可能保持句子的原本结构不变。
然后，直接返回纠错完成的句子，不要进行任何纠错的说明和结束。
请注意，返回句子需要使用<output> 纠错语句 </output>标签指出。

这里有些例子供你学习：{}

请学习并理解这些例子，然后完成下列句子的纠错。

<input> {} </input>:
"""

PROMPT_FOR_CHINESE_GEC = """
请纠正使用<input> 错误语句 </input>标签指出的句子中的所有语法错误和错别字。如果没有句子中没有错误，则直接返回原始句子。
请先将句子作为一个整体来理解，再进行纠错。纠错过程中，请尽可能保持句子的原本结构不变。
然后，直接返回纠错完成的句子，不要进行任何纠错的说明和结束。
请注意，返回句子需要使用<output> 纠错语句 </output>标签指出。

请完成下列句子的纠错。

<input> {} </input>:
"""


def get_examples(dataset_name, count):
    with open(os.path.join(os.getcwd(), "dataset", dataset_name, "trainset.json"), 'r', encoding='utf-8') as f:
        train_set = json.load(f)

    index_list = []
    while len(index_list) < count:
        random_index = random.randint(0, len(train_set) - 1)
        if random_index not in index_list:
            index_list.append(random_index)

    examples = []
    for i in index_list:
        examples.append(train_set[i])

    with open(os.path.join(os.getcwd(), "dataset", dataset_name, "testset.json"), 'r', encoding='utf-8') as f:
        test_set = json.load(f)

    return examples, test_set


def get_prompts(examples, test_set, dataset_name):
    icl_prompt = "\n"
    prompt_list = []
    if dataset_name in ['nlpcc18']:
        system_message = {"role": "system", "content": "你是语法纠错工具，能够检测并纠正文本中出现的所有语法错误。"}

        if len(examples) > 0:
            for example in examples:
                icl_prompt += ("<input> " + example["source_text"] + " </input>: <output> "
                               + example["target_text"] + "</output>\n\n")

            for i in range(len(test_set)):
                prompt = ICL_PROMPT_FOR_CHINESE_GEC.format(icl_prompt, test_set[i]["source_text"])
                prompt_message = {"role": "user", "content": prompt}
                prompt = [system_message, prompt_message]
                prompt_list.append(prompt)
        else:
            for i in range(len(test_set)):
                prompt = PROMPT_FOR_CHINESE_GEC.format(test_set[i]["source_text"])
                prompt_message = {"role": "user", "content": prompt}
                prompt = [system_message, prompt_message]
                prompt_list.append(prompt)

    elif dataset_name in ['conll14']:
        system_message = {"role": "system",
                          "content": "You are an English grammatical error correction tool "
                                     "and you should correct all the grammatical errors in the given sentences."}

        if len(examples) > 0:
            for example in examples:
                icl_prompt += ("<input> " + example["source_text"]
                               + " </input>: <output> " + example["target_text"] + " </output>\n\n")

            for i in range(len(test_set)):
                prompt = ICL_PROMPT_FOR_ENGLISH_GEC.format(icl_prompt, test_set[i]["source_text"])
                prompt_message = {"role": "user", "content": prompt}
                prompt = [system_message, prompt_message]
                prompt_list.append(prompt)
        else:
            for i in range(len(test_set)):
                prompt = PROMPT_FOR_ENGLISH_GEC.format(test_set[i]["source_text"])
                prompt_message = {"role": "user", "content": prompt}
                prompt = [system_message, prompt_message]
                prompt_list.append(prompt)

    return prompt_list


def run_prompts(model_name, dataset_name, example_count):
    directory = os.path.join(os.getcwd(), 'checkpoint', model_name + '-' + dataset_name)
    if not os.path.exists(directory):
        os.makedirs(directory)

    filename = "output_" + datetime.datetime.now().strftime("%Y-%m-%d-%H") + '.txt'
    f = open(os.path.join(directory, filename), 'w', encoding='utf-8')

    examples, test_set = get_examples(dataset_name, example_count)

    prompts = get_prompts(examples, test_set, dataset_name)

    # get model and tokenizer
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    for i in tqdm(range(len(prompts))):
        prompt = prompts[i]
        text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)

        model_inputs = tokenizer([text], return_tensors="pt").to(device)

        generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)

        generated_ids = [output_ids[len(input_ids):] for input_ids, output_ids
                         in zip(model_inputs.input_ids, generated_ids)]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        f.write(str(i) + '\n' + response + '\n\n')

