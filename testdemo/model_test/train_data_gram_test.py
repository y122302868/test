# -*- coding:utf-8 -*-
import os

import tensorflow as tf
from tensorflow_core.python.keras.optimizer_v2.adam import Adam

from testdemo.model_test import explore_data_test
from testdemo.model_test.explore_data_test import seg_words_per_sample
from testdemo.model_test.ngram import coll_data_gram_test, build_model_gram_test

root_path = os.path.join(os.path.abspath(os.path.dirname(__file__)).split('autohome_smart')[0], 'autohome_smart')


def train_ngram_model(data,
                      learning_rate=1e-4,
                      epochs=300,
                      batch_size=128,
                      layers=2,
                      units=64,
                      dropout_rate=0.2):
    """在给定数据集上训练n-gram模型。

    # 参数
        data: 训练和测试文本和标签的元组
        learning_rate: float, 训练模式的学习率。
        epochs: int, number of epochs.
        batch_size: int, 没批的数量
        layers: int, 模型中“密集”层的数量。
        units: int, 模型中稠密层的输出维数。
        dropout_rate: float: 输入到Dropout层的百分比。

    # Raises
        ValueError: 如果验证的数据的分类在训练数据中没有找到
    """
    # 获取数据
    (train_texts, train_labels), (val_texts, val_labels) = data

    # 验证测试标签与训练标签在同一范围内
    num_classes = explore_data_test.get_num_classes(train_labels)
    unexpected_labels = [v for v in val_labels if v not in range(num_classes)]
    if len(unexpected_labels):
        raise ValueError('Unexpected label values found in the validation set:'
                         ' {unexpected_labels}. Please make sure that the '
                         'labels in the validation set are in the same range '
                         'as training labels.'.format(
            unexpected_labels=unexpected_labels))

    # 进行分词
    seg_train_words_result = seg_words_per_sample(train_texts, os.path.join(root_path, 'jiebadict', 'stop_words.txt'),
                                                  os.path.join(root_path, 'jiebadict', 'dict.txt'))
    seg_test_words_result = seg_words_per_sample(val_texts, os.path.join(root_path, 'jiebadict', 'stop_words.txt'),
                                                 os.path.join(root_path, 'jiebadict', 'dict.txt'))

    # 向量化文本
    x_train, x_val = coll_data_gram_test.ngram_vectorize(
        seg_train_words_result, train_labels, seg_test_words_result)

    # 创建模型实例
    model = build_model_gram_test.mlp_model(layers=layers,
                                            units=units,
                                            dropout_rate=dropout_rate,
                                            input_shape=x_train.shape[1:],
                                            num_classes=num_classes)

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
        validation_data=(x_val, val_labels),
        verbose=2,  # Logs once per epoch.
        batch_size=batch_size)

    # 打印结果.
    history = history.history
    print('Validation accuracy: {acc}, loss: {loss}'.format(
        acc=history['val_acc'][-1], loss=history['val_loss'][-1]))

    # 保存模型.
    model.save(os.path.join(root_path, 'model', 'IMDb_mlp_model.h5'))
    # return history['val_acc'][-1], history['val_loss'][-1]
    return history
