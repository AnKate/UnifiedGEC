import json
import logging
import nltk
import random
import spacy
import string
import os

from openccpy import Opencc
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def is_english_char(word):
    english_char = set(string.ascii_lowercase)
    return all(char in english_char for char in word)


def is_chinese_char(word):
    for w in word:
        if u'\u4e00' > w or w > u'\u9fff':
            return False
    return True


class DataGenerator(object):

    def __init__(self, language):
        self.language = language
        self.vocab_path = os.path.join(os.getcwd(), "augmentation", "data", "vocab.txt")
        self.stopword_path = os.path.join(os.getcwd(), "augmentation", "data", "stop_words.txt")
        self.not_common_path = os.path.join(os.getcwd(), "augmentation", "data", "生僻字.txt")
        self.words_path = os.path.join(os.getcwd(), "augmentation", "data", "words.txt")
        self.vocab, self.stopwords, self.words = self.load_data()

    def load_data(self):
        stop_words, vocab = [], []
        with open(self.stopword_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                cont = line.strip('\n').strip()
                if is_chinese_char(cont):
                    if len(cont) < 2:
                        stop_words.append(cont)
                else:
                    stop_words.append(cont)

        non_common = [line.strip() for line in open(self.not_common_path, 'r', encoding='utf-8')]

        with open(self.vocab_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                cont = line.strip('\n').split('\t')
                if cont[0] not in stop_words and cont[0] not in non_common and int(cont[1]) > 5000:
                    if cont[0] != Opencc.to_simple(cont[0]):
                        vocab.append(Opencc.to_simple(cont[0]))
                    else:
                        vocab.append(cont[0])

        words = []
        with open(self.words_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.strip('\n')
                if len(line) <= 2:
                    words.append(line)

        return vocab, stop_words, words

    def data_generate(self, input_data):
        noise_types = ["insert", "delete", "replace", "swap"]
        augmented_data = []

        for item in input_data:
            case = {"id": item[0], "source_text": item[1], "target_text": item[2]}
            augmented_data.append(case)

        offset = len(augmented_data)

        for item in tqdm(input_data):
            noise_type = random.choice(noise_types)
            augmented_text = self.add_word_level_noise(item[1], noise_type)
            new_case = {"id": item[0] + offset, "source_text": augmented_text, "target_text": item[2]}
            augmented_data.append(new_case)

        return augmented_data

    def add_word_level_noise(self, text, noise_type, num_errors=1):
        if self.language == 'en':
            nlp = spacy.load('en_core_web_sm')
            words_list = [word.text for word in nlp.vocab if not word.is_stop]
        else:
            words_list = self.vocab

        for _ in range(num_errors):
            if self.language == 'en':
                words = nltk.word_tokenize(text)
            else:
                words = []
                current_segment = ''
                for word in text:
                    if is_chinese_char(word):
                        if current_segment:
                            words.append(current_segment)
                            current_segment = ''
                        words.append(word)
                    else:
                        current_segment += word
                if current_segment:
                    words.append(current_segment)
                # print(words)

            num_words = len(words)
            if num_words == 0:
                return text

            index = random.randint(0, num_words - 1)

            if self.language == 'zh':
                random_word = ''
                while not is_english_char(random_word):
                    random_word = random.choice(words_list)
            else:
                random_word = random.choice(words_list)
            if noise_type == "insert":
                words.insert(index, random_word)
            elif noise_type == "delete" and num_words > 1:
                words.pop(index)
            elif noise_type == "replace":
                words[index] = random_word
            elif noise_type == "swap" and num_words > 1:
                if index + 1 < num_words:
                    words[index], words[index + 1] = words[index + 1], words[index]

            text = ''.join(words)

        return text


def noise_pattern(origin_dataset, language, percentage=0.1):
    if language == 'en':
        nltk.download('punkt')

    dataset_len = int(len(origin_dataset) * percentage)

    original_data = []
    for idx, data in enumerate(origin_dataset[:dataset_len]):
        source_text = data['source_text']
        target_text = data['target_text']

        original_data.append([idx, source_text, target_text])

    generator = DataGenerator(language)
    augmented_dataset = generator.data_generate(original_data)

    return augmented_dataset

