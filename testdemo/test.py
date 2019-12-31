# -*- coding:utf-8 -*-
import os
from testdemo.model_test.explore_data_test import load_imdb_sentiment_analysis_dataset, seg_words_per_sample, \
    get_num_words_per_sample


# 获取跟路径
root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('autohome_smart')[0], 'autohome_smart')

# 准备数据
((train_texts, train_labels), (test_texts, test_labels)) = load_imdb_sentiment_analysis_dataset(root_path)

seg_train_words_result = seg_words_per_sample(train_texts, os.path.join(root_path, 'jiebadict', 'stop_words.txt'),
                                                  os.path.join(root_path, 'jiebadict', 'dict.txt'))

w = len(train_texts)/get_num_words_per_sample(seg_train_words_result)
print(w)