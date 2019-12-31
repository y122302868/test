# -*- coding:utf-8 -*-


from tensorflow_core.python.keras.api.keras import models
from tensorflow_core.python.keras.layers.convolutional import SeparableConv1D

from tensorflow_core.python.keras.layers.core import Dense, Dropout
from tensorflow_core.python.keras.layers.embeddings import Embedding
from tensorflow_core.python.keras.layers.pooling import MaxPooling1D, GlobalAveragePooling1D

from testdemo.model_test.utils.choose_last_activate_test import _get_last_layer_units_and_activation


def sepcnn_model(blocks,
                 filters,
                 kernel_size,
                 embedding_dim,
                 dropout_rate,
                 pool_size,
                 input_shape,
                 num_classes,
                 num_features,
                 use_pretrained_embedding=False,
                 is_embedding_trainable=False,
                 embedding_matrix=None):
    """创建一个separable CNN 模型实例

    # 参数
        blocks: int, 模型中成对的sepCNN和池块数。
        filters: int, 层的输出维度。
        kernel_size: int, 卷积窗口的长度。
        embedding_dim: int, 嵌入向量的维数。
        dropout_rate: float, 输入到Dropout层的百分比。
        pool_size: int, 用于在MaxPooling层上降低输入规模的因子。
        input_shape: tuple, 模型输入的形状。
        num_classes: int, 输出类的数量。
        num_features: int, 字数(嵌入输入维数)。
        use_pretrained_embedding: bool, 如果预先训练的嵌入是打开的，则为true
        is_embedding_trainable: bool, 如果嵌入层是可训练的，则为true
        embedding_matrix: dict, 嵌入系数字典

    # 返回
        一个sepCNN模型实例。
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()

    # 添加嵌入层。如果使用预训练的嵌入，则向嵌入层添加权重，并设置trainable来输入is_embedding_trainable标志。
    if use_pretrained_embedding:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0],
                            weights=[embedding_matrix],
                            trainable=is_embedding_trainable))
    else:
        model.add(Embedding(input_dim=num_features,
                            output_dim=embedding_dim,
                            input_length=input_shape[0]))

    for _ in range(blocks - 1):
        model.add(Dropout(rate=dropout_rate))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(SeparableConv1D(filters=filters,
                                  kernel_size=kernel_size,
                                  activation='relu',
                                  bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform',
                                  padding='same'))
        model.add(MaxPooling1D(pool_size=pool_size))

    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(SeparableConv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform',
                              padding='same'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(rate=dropout_rate))
    model.add(Dense(op_units, activation=op_activation))
    return model



