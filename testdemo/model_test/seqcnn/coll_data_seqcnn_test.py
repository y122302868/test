# -*- coding:utf-8 -*-
from keras_preprocessing import sequence
from keras_preprocessing import text

# 向量化参数
# 限制特性的数量。我们使用最顶级的2万
TOP_K = 20000

# 限制文本序列的长度。超过此长度的序列将被截断。
MAX_SEQUENCE_LENGTH = 500


def sequence_vectorize(train_texts, val_texts):
    """将文本向量化为序列向量

    一个文本 = 一个固定长度的序列向量

    # 参数
        train_texts: 列表，训练文本字符串
        val_texts: 列表，验证文本字符串

    # 返回
        x_train, x_val, word_index: 向量化的训练和验证文本和单词索引字典
    """
    # 用训练文本创造词汇
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # 向量化训练和验证文本
    x_train = tokenizer.texts_to_sequences(train_texts)
    x_val = tokenizer.texts_to_sequences(val_texts)

    # 得到最大的序列长度
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # 将序列长度固定为最大值。短于长度的序列在开始时填充，长于长度的序列在开始时截断。
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, x_val, tokenizer.word_index


def sequence_vectorize(train_texts):
    """将文本向量化为序列向量

    一个文本 = 一个固定长度的序列向量

    # 参数
        train_texts: 列表，训练文本字符串
        val_texts: 列表，验证文本字符串

    # 返回
        x_train, x_val, word_index: 向量化的训练和验证文本和单词索引字典
    """
    # 用训练文本创造词汇
    tokenizer = text.Tokenizer(num_words=TOP_K)
    tokenizer.fit_on_texts(train_texts)

    # 向量化训练和验证文本
    x_train = tokenizer.texts_to_sequences(train_texts)
    # x_val = tokenizer.texts_to_sequences(val_texts)

    # 得到最大的序列长度
    max_length = len(max(x_train, key=len))
    if max_length > MAX_SEQUENCE_LENGTH:
        max_length = MAX_SEQUENCE_LENGTH

    # 将序列长度固定为最大值。短于长度的序列在开始时填充，长于长度的序列在开始时截断。
    x_train = sequence.pad_sequences(x_train, maxlen=max_length)
    # x_val = sequence.pad_sequences(x_val, maxlen=max_length)
    return x_train, tokenizer.word_index
