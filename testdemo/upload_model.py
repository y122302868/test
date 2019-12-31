# -*- coding:utf-8 -*-

import os
import os.path as osp
from tensorflow_core.python.keras.models import load_model
import tensorflow as tf

# 先将h5模型转换为pb
from testdemo.model_test.explore_data_test import seg_words_per_sample, load_imdb_sentiment_analysis_dataset
from testdemo.model_test.ngram.coll_data_gram_test import ngram_vectorize
from testdemo.model_test.utils.pb_util import h5_to_pb, load_pb

root_path = osp.join(osp.abspath(osp.dirname(__file__)).split('autohome_smart')[0], 'autohome_smart')
# 路径参数
input_path = osp.join(root_path, 'model')
weight_file = 'IMDb_mlp_model.h5'
weight_file_path = osp.join(input_path, weight_file)
output_graph_name = weight_file[:-3] + '.pb'

# 输出路径
output_dir = osp.join(root_path, "model_pb")
# 加载模型
h5_model = load_model(weight_file_path)
# h5_to_pb(h5_model, output_dir=output_dir, model_name=output_graph_name)
# print('model save h5 to pb')
#
#
# # 加载pb模型进行预测
# load_pb(osp.join(output_dir, output_graph_name))

# 获取数据
((train_texts, train_labels), (test_texts, test_labels)) = load_imdb_sentiment_analysis_dataset(root_path)
seg_train_word = seg_words_per_sample(train_texts, os.path.join(root_path, 'jiebadict', 'stop_words.txt'),
                                                  os.path.join(root_path, 'jiebadict', 'dict.txt'))
seg_word = seg_words_per_sample(['S80L的后排头部空间较A6L的要小一点'], os.path.join(root_path, 'jiebadict', 'stop_words.txt'),
                                                  os.path.join(root_path, 'jiebadict', 'dict.txt'))
# 向量化
x_train, x_val = ngram_vectorize(seg_train_word, train_labels, seg_word)
predictions = h5_model.predict(x_val)
print(predictions)