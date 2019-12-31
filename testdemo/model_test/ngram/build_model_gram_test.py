# -*- coding:utf-8 -*-

from tensorflow_core.python.keras.api.keras import models
from tensorflow_core.python.keras.layers.core import Dense, Dropout
from testdemo.model_test.utils.choose_last_activate_test import _get_last_layer_units_and_activation


def mlp_model(layers, units, dropout_rate, input_shape, num_classes):
    """创建一个多层感知器模型的模型

    # 参数
        layers: int，模型中密集层的数量。
        units: int, 层的输出维度
        dropout_rate: float, 输入到Dropout层的百分比。
        input_shape: tuple, 模型输入的形状。
        num_classes: int, 输出类的数量。

    # 返回
        一个 MLP 模型的初始化
    """
    op_units, op_activation = _get_last_layer_units_and_activation(num_classes)
    model = models.Sequential()
    model.add(Dropout(rate=dropout_rate, input_shape=input_shape))

    for _ in range(layers - 1):
        model.add(Dense(units=units, activation='relu'))
        model.add(Dropout(rate=dropout_rate))

    model.add(Dense(units=op_units, activation=op_activation))
    return model
