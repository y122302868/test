# -*- coding:utf-8 -*-
import os
import os.path as osp
import tensorflow.compat.v1 as tf
from tensorflow_core.python.keras import backend as K
from tensorflow_core.python.platform import gfile

'''
将h5模型转换为pb格式
'''


def h5_to_pb(h5_model, output_dir, model_name, out_prefix="output_", log_tensorboard=True):
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
    out_nodes = []
    for i in range(len(h5_model.outputs)):
        out_nodes.append(out_prefix + str(i + 1))
        tf.identity(h5_model.outputs[i], out_prefix + str(i + 1))
    sess = K.get_session()
    from tensorflow_core.python.framework import graph_util, graph_io
    init_graph = sess.graph.as_graph_def()
    main_graph = graph_util.convert_variables_to_constants(sess, init_graph, out_nodes)
    graph_io.write_graph(main_graph, output_dir, name=model_name, as_text=False)
    if log_tensorboard:
        from tensorflow_core.python.tools import import_pb_to_tensorboard
        import_pb_to_tensorboard.import_to_tensorboard(osp.join(output_dir, model_name), output_dir)


def load_pb(pb_file_path):
    sess = tf.Session()
    with gfile.FastGFile(pb_file_path, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        sess.graph.as_graph_def(graph_def, name='')

    print(sess.run('b:0'))
    # 输入
    input_x = sess.graph.get_tensor_by_name('x:0')
    input_y = sess.graph.get_tensor_by_name('y:0')
    # 输出
    op = sess.graph.get_tensor_by_name('op_to_store:0')
    # 预测结果
    ret = sess.run(op, {input_x: 3, input_y: 4})
    print(ret)