# -*- coding:utf-8 -*-


def _get_last_layer_units_and_activation(num_classes):
    """获取最后一个网络层的# units和激活函数。

    # 参数
        num_classes: int，类的数量。

    # 返回
        单位,激活函数
    """
    if num_classes == 2:
        activation = 'sigmoid'
        units = 1
    else:
        activation = 'softmax'
        units = num_classes
    return units, activation
