import re
import os
import math
from collections import Counter, defaultdict

import nltk
from nltk.corpus import reuters

# 下载路透社语料库和 punkt 分词器 (准确率能提高 10 左右)
nltk.download('reuters')
nltk.download('punkt')

def read_lines(file_path):
    with open(file_path, 'r') as f:
        return f.readlines()

def read_vocabulary(file_path):
    with open(file_path, 'r') as file:
        return set(file.read().split())

# 从数据中获取错误与正确单词对
def extract_pairs(data):
    pairs = []
    for line in data:
        parts = line.strip().split()
        if len(parts) > 3:
            pairs.append((parts[2], parts[3]))
    return pairs

# 构建通道模型
def build_channel_model(pairs):
    error_model = defaultdict(lambda: defaultdict(int))
    for error, correct in pairs:
        error_model[correct][error] += 1
    return error_model

# 生成拼写错误的候选纠正词
def generate_candidates(word, vocabulary):
    letters = 'abcdefghijklmnopqrstuvwxyz'
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [L + R[1:] for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
    replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
    inserts = [L + c + R for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts) & vocabulary

# 生成候选纠正词的回退方法
def generate_candidates_fallback(word, vocabulary):
    letters = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    candidates = set()
    for i in range(len(word)):
        candidate = word[:i] + word[i + 1:]
        candidates.add(candidate)
        for c in letters:
            candidates.add(word[:i] + c + word[i + 1:])
            candidates.add(word[:i] + c + word[i:])
        for j in range(i + 2, len(word)):
            candidates.add(word[:i] + word[j] + word[i + 1:j] + word[i] + word[j + 1:])
    return candidates & vocabulary

# 综合候选词生成方法
def generate_candidates_with_fallback(word, vocabulary):
    candidates = generate_candidates(word, vocabulary)
    if not candidates:
        candidates = generate_candidates_fallback(word, vocabulary)
    return candidates

# 训练n-gram语言模型
def build_ngram_model(corpus, n, vocab):
    def extract_ngrams(tokens, n):
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    ngrams = Counter()
    for sentence in corpus:
        tokens = ['<s>'] * (n - 1) + [word for word in sentence.split() if word in vocab] + ['</s>']
        ngrams.update(extract_ngrams(tokens, n))
    total_ngrams = sum(ngrams.values())
    return ngrams, total_ngrams

def get_corpus_from_reuters():
    return reuters.sents()

# 计算句子的概率
def calculate_sentence_probability(sentence, ngram_model, total_ngrams, n):
    tokens = ['<s>'] * (n - 1) + sentence.split() + ['</s>']
    score = 0
    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngram_count = ngram_model[ngram] + 1
        context_count = ngram_model[ngram[:-1]] + total_ngrams
        score += math.log(ngram_count / context_count)
    return score

# 分离单词与标点
def split_word_punctuation(word):
    punctuations = '",.!?;:\''
    for punctuation in punctuations:
        if word.endswith(punctuation):
            return word[:-1], punctuation
    return word, ''

# 分离缩略形式
def split_contraction(word):
    contractions = ["'ll", "'s", "n't", "'ve", "'re", "'d", "'m"]
    for contraction in contractions:
        if word.endswith(contraction):
            return word[:-len(contraction)], contraction
    return word, ''

# 使用回退方法进行句子纠正
def correct_sentence_fallback(sentence, channel_model, ngram_model, total_ngrams, n, vocabulary):
    def replace_word_in_sentence(sentence, old_word, new_word):
        pattern = re.compile(r'\b' + re.escape(old_word) + r'\b')
        return pattern.sub(new_word, sentence)

    words = re.findall(r'\w+|[^\w\s]', sentence)
    corrected_words = []
    for word in words:
        if re.match(r'\w+', word):
            if word not in vocabulary:
                candidates = generate_candidates_with_fallback(word, vocabulary)
                if not candidates:
                    corrected_words.append(word)
                else:
                    best_candidate = max(candidates, key=lambda w: calculate_sentence_probability(
                        replace_word_in_sentence(sentence, word, w), ngram_model, total_ngrams, n))
                    corrected_words.append(best_candidate)
            else:
                corrected_words.append(word)
        else:
            corrected_words.append(word)
    return ''.join(corrected_words)

def correct_sentence(sentence, channel_model, ngram_model, total_ngrams, n, vocabulary):
    def replace_word_in_sentence(sentence, old_word, new_word):
        pattern = re.compile(r'\b' + re.escape(old_word) + r'\b')
        return pattern.sub(new_word, sentence)

    words = re.findall(r'\S+|\s+', sentence)
    corrected_words = []
    for word in words:
        if word in vocabulary:
            corrected_words.append(word)
        else:
            main_word_first, punctuation = split_word_punctuation(word)
            if re.match(r'\w+', main_word_first):
                main_word, contraction = split_contraction(main_word_first)
                if main_word not in vocabulary:
                    candidates = generate_candidates_with_fallback(main_word, vocabulary)
                    if not candidates:
                        corrected_words.append(
                            correct_sentence_fallback(word, channel_model, ngram_model, total_ngrams, n, vocabulary))
                    else:
                        best_candidate = max(candidates, key=lambda w: calculate_sentence_probability(
                            replace_word_in_sentence(sentence, main_word, w), ngram_model, total_ngrams, n))
                        corrected_words.append(best_candidate + contraction + punctuation)
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
    return ''.join(corrected_words)

def main():
    vocabulary = read_vocabulary('vocab.txt')
    train_data = read_lines('text.txt')
    error_pairs = extract_pairs(train_data)
    channel_model = build_channel_model(error_pairs)
    test_data = read_lines('testData.txt')
    corpus = get_corpus_from_reuters()
    corpus = [' '.join(sentence) for sentence in corpus]
    n = 3
    ngram_model, total_ngrams = build_ngram_model(corpus, n, vocabulary)
    results = []

    for line in test_data:
        parts = line.strip().split('\t')
        if len(parts) == 3:
            id, num_errors, sentence = parts
            corrected_sentence = correct_sentence(sentence, channel_model, ngram_model, total_ngrams, n, vocabulary)
            results.append(f"{id}\t{corrected_sentence}")

    with open('result.txt', 'w') as f:
        for result in results:
            f.write(result + '\n')

    os.system('python eval.py')


if __name__ == "__main__":
    main()
