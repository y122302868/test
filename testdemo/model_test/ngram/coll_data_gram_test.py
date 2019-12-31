# -*- coding:utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

# 向量化参数
# 标记文本的n-gram的大小范围
NGRAM_RANGE = (1, 2)

# 限制特性的数量。我们使用最顶级的2万
TOP_K = 20000

# 文本被分成单词还是字符n-grams
# 'word', 'char'.
TOKEN_MODE = 'word'

# 令牌出现的次数小于文档/语料库的将被丢弃
MIN_DOCUMENT_FREQUENCY = 2


def ngram_vectorize(train_texts, train_labels, val_texts):
    """将文本向量化为n-gram

    1个文本 = 1个tf-idf向量的单字格长度+双字格长度。

    # 参数
        train_texts: 列表，训练文本字符串。
        train_labels: np.ndarray, 培训标签
        val_texts: 列表，验证文本字符串。

    # 返回
        x_train, x_val: 向量化的培训和验证文本
    """
    # 创建关键字参数传递给'tf-idf'矢量转换器
    kwargs = {
        'ngram_range': NGRAM_RANGE,  # Use 1-grams + 2-grams.
        'dtype': 'int32',
        'strip_accents': 'unicode',
        'decode_error': 'replace',
        'analyzer': TOKEN_MODE,  # 将文本分割为单词标记
        'min_df': MIN_DOCUMENT_FREQUENCY,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # 从训练数据中学习词汇并且向量化
    x_train = vectorizer.fit_transform(train_texts)
    # 矢量化验证文本
    x_val = vectorizer.transform(val_texts)

    # 选择向量化特征的顶部'k'。
    selector = SelectKBest(f_classif, k=min(TOP_K, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val

