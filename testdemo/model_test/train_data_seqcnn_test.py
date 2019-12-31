# -*- coding:utf-8 -*-
import os

import tensorflow as tf
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from testdemo.model_test import explore_data_test
from testdemo.model_test.explore_data_test import seg_words_per_sample
from testdemo.model_test.seqcnn import coll_data_seqcnn_test, build_model_seqcnn_test

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('autohome_smart')[0], 'autohome_smart')
FLAGS = None

# 限制特性的数量，我们使用最顶级的20000
TOP_K = 20000


def train_sequence_model(data,
                         learning_rate=1e-3,
                         epochs=1000,
                         batch_size=128,
                         blocks=2,
                         filters=64,
                         dropout_rate=0.2,
                         embedding_dim=200,
                         kernel_size=3,
                         pool_size=3):
    """在给定数据集上训练序列模型。

    # 参数
        data: 训练和测试文本和标签的元组
        learning_rate: float, 训练模式的学习率。
        epochs: int, number of epochs.
        batch_size: int, 每批的数量
        blocks: int, 成对的sepCNN和池块的模型
        filters: int, 模型中sepCNN层的输出尺寸
        dropout_rate: float: 输入到Dropout层的百分比。
        embedding_dim: int, 嵌入向量的维数。
        kernel_size: int, 卷积窗口的长度。
        pool_size: int, 用于降低MaxPooling层输入规模的因数。


    # Raises
        ValueError: 如果验证的数据的分类在训练数据中没有找到
    """
    # 获取数据
    # (train_texts, train_labels), (val_texts, val_labels) = data
    train_texts, train_labels = data

    # 验证测试标签与训练标签在同一范围内
    num_classes = explore_data_test.get_num_classes(train_labels)
    # unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    # if len(unexpected_labels):
    #     raise ValueError('Unexpected label values found in the validation set:'
    #                      ' {unexpected_labels}. Please make sure that the '
    #                      'labels in the validation set are in the same range '
    #                      'as training labels.'.format(
    #         unexpected_labels=unexpected_labels))

    # 进行分词
    seg_train_words_result = seg_words_per_sample(train_texts, os.path.join(root_path, 'jiebadict', 'stop_words.txt'),
                                                  os.path.join(root_path, 'jiebadict', 'dict.txt'))
    # seg_test_words_result = seg_words_per_sample(val_texts, os.path.join(root_path, 'jiebadict', 'stop_words.txt'),
    #                                              os.path.join(root_path, 'jiebadict', 'dict.txt'))

    # 向量化文本
    # x_train, x_val, word_index = coll_data_seqcnn_test.sequence_vectorize(
    #     seg_train_words_result, seg_test_words_result)
    x_train, word_index = coll_data_seqcnn_test.sequence_vectorize(seg_train_words_result)

    # 特征数为嵌入输入维数。为保留的索引添加1
    num_features = min(len(word_index) + 1, TOP_K)

    # 创建模型实例
    model = build_model_seqcnn_test.sepcnn_model(blocks=blocks,
                                                 filters=filters,
                                                 kernel_size=kernel_size,
                                                 embedding_dim=embedding_dim,
                                                 dropout_rate=dropout_rate,
                                                 pool_size=pool_size,
                                                 input_shape=x_train.shape[1:],
                                                 num_classes=num_classes,
                                                 num_features=num_features)

    # 使用学习参数编译模型。
    if num_classes == 2:
        loss = 'binary_crossentropy'
    else:
        loss = 'sparse_categorical_crossentropy'
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss=loss, metrics=['acc'])

    # 创建回调，以便在验证失败时尽早停止。如果连续两次没有减少，就停止训练。
    callbacks = [tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=2)]

    # 训练和验证模型。
    history = model.fit(
        x_train,
        train_labels,
        epochs=epochs,
        callbacks=callbacks,
        # validation_data=(x_val, val_labels)
        validation_split=0.2,
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)

    # 打印结果.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # 保存模型.
    model.save(os.path.join(root_path, 'model', 'rotten_tomatoes_sepcnn_model.h5'))
    # return history['val_acc'][-1], history['val_loss'][-1]
    return history
