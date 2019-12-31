# -*- coding:utf-8 -*-
import os
from testdemo.model_test.explore_data_test import load_imdb_sentiment_analysis_dataset
from testdemo.model_test.utils.plot_accuracy_loss_test import make_plot


# 获取跟路径
from testdemo.model_test.train_data_seqcnn_test import train_sequence_model

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('autohome_smart')[0], 'autohome_smart')

# 准备数据
# ((train_texts, train_labels), (test_texts, test_labels)) = load_imdb_sentiment_analysis_dataset(root_path)
train_texts, train_labels = load_imdb_sentiment_analysis_dataset(root_path)

# 获取类型数量
label_list = []
for label in train_labels:
    if label not in label_list:
        label_list.append(label)

# 训练模型
# history = train_ngram_model(((train_texts, train_labels), (test_texts, test_labels)))
# history = train_sequence_model(((train_texts, train_labels), (test_texts, test_labels)))
history = train_sequence_model(((train_texts, train_labels), (test_texts, test_labels)))
# make_plot(history, 1)

make_plot(history, 2)
