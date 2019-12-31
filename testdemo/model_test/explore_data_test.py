import os
import random

import jieba
import numpy as np
from matplotlib import pyplot as plt
from jieba import cut

"""加载汽车评论分析数据集

    # 参数
        data_path: 字符串，数据目录的路径。
        seed: int，种子随机化。

    # 返回
        训练和验证的一个元组。
        训练样本数目: 25000
        试验样品数量: 25000
        数量的类别: 2 (0 - 负面, 1 - 正面)
"""


def load_imdb_sentiment_analysis_dataset(data_path, seed=123):
    imdb_data_path = os.path.join(data_path, 'labeldetail')

    # 加载培训数据
    train_texts = []
    train_labels = []
    train_label_total = []
    # 查看训练数据的所有分类
    train_path = os.path.join(imdb_data_path, 'train')
    for filename in sorted(os.listdir(train_path)):
        train_label_total.append(filename)
    for category in train_label_total:
        train_path = os.path.join(imdb_data_path, 'train', category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname), encoding='utf-8') as f:
                    for line in f:
                        # 过滤符号并判空
                        line = line.replace('<B>', '')
                        line = line.replace('</B>', '')
                        line = ''.join(filter(lambda word: len(word) > 1, cut(line)))
                        if len(line) > 0 and line not in train_texts:
                            train_texts.append(line)
                            train_labels.append(train_label_total.index(category))
    # 加载验证数据
    # test_texts = []
    # test_labels = []
    # test_label_total = []
    # test_path = os.path.join(imdb_data_path, 'test')
    # for filename in sorted(os.listdir(test_path)):
    #     test_label_total.append(filename)
    # for category in test_label_total:
    #     test_path = os.path.join(imdb_data_path, 'test', category)
    #     for fname in sorted(os.listdir(test_path)):
    #         if fname.endswith('.txt'):
    #             with open(os.path.join(test_path, fname), encoding='utf-8') as f:
    #                 for line in f:
    #                     # 过滤符号并判空
    #                     line = line.replace('<B>', '')
    #                     line = line.replace('</B>', '')
    #                     line = ''.join(filter(lambda word: len(word) > 1, cut(line)))
    #                     if len(line) > 0 and line not in test_texts:
    #                         test_texts.append(line)
    #                         test_labels.append(test_label_total.index(category))

    # 重新整理培训数据和标签
    random.seed(seed)
    random.shuffle(train_texts)
    random.seed(seed)
    random.shuffle(train_labels)
    return train_texts, np.array(train_labels)
    # return ((train_texts, np.array(train_labels)),
    #         (test_texts, np.array(test_labels)))


"""返回给定语料库中每个样本的单词数的中位数

    # 参数
        sample_texts: 列表,示例文本

    # 返回
        int, 每个样本的单词数中位数
    """


def seg_words_per_sample(sample_texts, stop_word_path, local_word_path):
    # 把停用词做成字典
    stopwords = {}
    if len(stop_word_path) > 0:
        fstop = open(stop_word_path, 'r', encoding='utf-8', errors='ignore')
        for eachWord in fstop:
            stopwords[eachWord.strip()] = eachWord.strip()  # 停用词典
        fstop.close()

    # 结巴分词加载自定义词库
    if len(local_word_path) > 0:
        jieba.load_userdict(local_word_path)
    seg_word_result = []
    for i in range(len(sample_texts)):
        sentence = sample_texts[i]
        # 去前后的空格
        sentence = sentence.strip()
        # 去标点符号
        sentence = jieba.re.sub(r"[\s+\.\!\/_,$%^*()?;；:-【】+\"\']+|[+——！，;:。？、~@#￥%……&*（）]+", " ", sentence)

        seg_list = jieba.cut(sentence, cut_all=False, HMM=True)
        seg_word = ''
        for word in seg_list:
            if word not in stopwords:
                if not word.isspace():
                    seg_word += word
                    seg_word += ' '
        seg_word = seg_word.strip()
        if len(seg_word) > 0:
            seg_word_result.append(seg_word)
    return seg_word_result


"""返回给定语料库中每个样本的单词数的中位数

    # 参数
        sample_texts: 列表,示例文本

    # 返回
        int, 每个样本的单词数中位数
    """


def get_num_words_per_sample(sample_texts):
    num_words = [len(s.split()) for s in sample_texts]
    return np.median(num_words)


"""绘制样本长度分布。

    # 参数
        samples_texts: 列表,示例文本。
    """


def plot_sample_length_distribution(sample_texts):
    plt.hist([len(s) for s in sample_texts], 50)
    plt.xlabel('Length of a sample')
    plt.ylabel('Number of samples')
    plt.title('Sample length distribution')
    plt.show()


def get_num_classes(sample_label):
    num_classes = []
    for label in sample_label:
        if label not in num_classes:
            num_classes.append(label)

    return len(num_classes)
