# -- encoding:utf-8 --
"""
Create by ibf on 19/1/10
"""

import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

tf.set_random_seed(28)


def create_model(images):
    """
    基于传入的数据进行网络的构建
    :param images:
    :return:
    """
    # 定义网络结构(Input -> Conv -> ReLu -> Pooling -> FC  -> ReLu -> FC)
    with tf.variable_scope('net', initializer=tf.random_normal_initializer(0.0, 0.1)):
        with tf.variable_scope("input"):
            # 对输入的数据做一个前期转换的处理（这里可以加入一些图像预处理相关的操作）
            print(images.get_shape())
            net = tf.reshape(images, shape=[-1, 28, 28, 1])
            tf.summary.image("img", net)
            print(net.get_shape())
        with tf.variable_scope("Conv1"):
            """
            def conv2d(input, filter, strides, padding, use_cudnn_on_gpu=True, data_format="NHWC", name=None):
                功能：对输入的图像数据做一个卷积的操作
                参数：
                    input：给定对那个tensor对象所表示的图像数据做卷积操作，要求格式为: 默认为[batch_size, height, width, channels]，具体格式和参数data_format有关，当data_format为NHWC的时候，input要求的格式为默认格式；当data_format值为NCHW，input要求的格式为：[batch_size, channels, height, width]
                    filter: 卷积核也就是权重系数w，格式要求是一个4维的Tensor对象：[height, width, in_channels, out_channels] -> height指定卷积过程中窗口的高度，width指定卷积过程中窗口的宽度信息，in_channels表示输入的数据中有多少个feature map，也就是有多少个channels通道；out_channels表述的是经过当前层次的卷积输出的feature map数量，也就是卷积核数量、产生的新数据的通道数量。
                    strides: 给定窗口滑动的步长，格式和data_format有关，当为NHWC的时候，格式为：[batch, in_height, in_width, in_channels]；当为NCHW的时候，格式为：[batch, in_channels, in_height, in_width]; 其中batch和in_channels必须为1，in_height和in_width指定在输入数据的高度和宽度每次滑动多少.
                    padding: 当输入数据的height以及width和步长、窗口大小没法整除的时候，指定做什么操作，可选值为："SAME"表示进行最小填充, "VALID"表示对于没法计算的像素直接删除不计算。NOTE: 当padding值值SAME的时候，并且strides步长为1的时候，那么经过卷积之后的图像是不发生变化的，会做一个填充的操作。
                    use_cudnn_on_gpu: 如果你安装的GPU版本的tensorflow。那么是否启动cudnn的加速。NOTE:不是所有的GPU都支持cudnn加速的。
                    
            """
            w = tf.get_variable(name='w', shape=[3, 3, 1, 20])
            b = tf.get_variable(name='b', shape=[20])
            net = tf.nn.conv2d(input=net, filter=w, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, b)

            shape = net.get_shape()
            print(shape)
            for k in range(shape[-1]):
                tf.summary.image("img_{}".format(k),
                                 tf.reshape(net[:, :, :, k], shape=[-1, shape[1], shape[2], 1]))
        with tf.variable_scope("ReLu1"):
            net = tf.nn.relu(net)

            shape = net.get_shape()
            print(shape)
            for k in range(shape[-1]):
                tf.summary.image("img_{}".format(k),
                                 tf.reshape(net[:, :, :, k], shape=[-1, shape[1], shape[2], 1]))
        with tf.variable_scope("Conv2"):
            w = tf.get_variable(name='w', shape=[3, 3, 20, 40])
            b = tf.get_variable(name='b', shape=[40])
            net = tf.nn.conv2d(input=net, filter=w, strides=[1, 1, 1, 1], padding='SAME')
            net = tf.nn.bias_add(net, b)

            shape = net.get_shape()
            print(shape)
            for k in range(shape[-1]):
                tf.summary.image("img_{}".format(k),
                                 tf.reshape(net[:, :, :, k], shape=[-1, shape[1], shape[2], 1]))
        with tf.variable_scope("ReLu2"):
            net = tf.nn.relu(net)

            shape = net.get_shape()
            print(shape)
            for k in range(shape[-1]):
                tf.summary.image("img_{}".format(k),
                                 tf.reshape(net[:, :, :, k], shape=[-1, shape[1], shape[2], 1]))
        with tf.variable_scope("Pooling1"):
            """
            def max_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
            def avg_pool(value, ksize, strides, padding, data_format="NHWC", name=None):
              value：给定需要对那个数据进行池化的操作，要求格式为4维的Tensor对象；形状为： 默认为[batch_size, height, width, channels]，具体格式和参数data_format有关，当data_format为NHWC的时候，input要求的格式为默认格式；当data_format值为NCHW，input要求的格式为：[batch_size, channels, height, width]
              ksize：给定窗口大小，是一个4元组/数组，值格式为：data_format为NHWC[batch, in_height, in_width, in_channels]或者data_format值为NCHW[batch, in_channels, in_height, in_width]; 要求batch和in_channels必须为1，in_height和in_width指定的就是窗口的高度和宽度
              strides： 给定窗口滑动的步长，格式和data_format有关，当为NHWC的时候，格式为：[batch, in_height, in_width, in_channels]；当为NCHW的时候，格式为：[batch, in_channels, in_height, in_width]; 其中batch和in_channels必须为1，in_height和in_width指定在输入数据的高度和宽度每次滑动多少.
              padding: 当输入数据的height以及width和步长、窗口大小没法整除的时候，指定做什么操作，可选值为："SAME"表示进行最小填充, "VALID"表示对于没法计算的像素直接删除不计算。NOTE: 当padding值值SAME的时候，并且strides步长为1的时候，那么经过池化之后的图像是不发生变化的，会做一个填充的操作。
            """
            net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

            shape = net.get_shape()
            print(shape)
            for k in range(shape[-1]):
                tf.summary.image("img_{}".format(k),
                                 tf.reshape(net[:, :, :, k], shape=[-1, shape[1], shape[2], 1]))
        with tf.variable_scope("Reshape"):
            # 因为卷积这里不改变图像的大小，只通过一个池化改变了一半
            shape = net.get_shape()
            size = shape[1] * shape[2] * shape[3]
            net = tf.reshape(net, shape=(-1, size))
            print(net.get_shape())
        with tf.variable_scope("FC1"):
            w = tf.get_variable(name='w', shape=[size, 500])
            b = tf.get_variable(name='b', shape=[500])
            net = tf.nn.bias_add(tf.matmul(net, w), b)
            print(net.get_shape())
        with tf.variable_scope("ReLu3"):
            net = tf.nn.relu(net)
            print(net.get_shape())
        with tf.variable_scope("FC2"):
            w = tf.get_variable(name='w', shape=[500, 10])
            b = tf.get_variable(name='b', shape=[10])
            net = tf.add(tf.matmul(net, w), b)
            print(net.get_shape())
        with tf.variable_scope("Softmax"):
            p = tf.nn.softmax(net)
            y_predict = tf.argmax(p, 1)

    # 返回最终的输出
    return p, y_predict


def create_loss(labels, logistic):
    """
    基于预测值和实际值构建损失函数
    :param labels:
    :param logistic:
    :return:
    """
    with tf.variable_scope("loss"):
        loss = -tf.reduce_mean(tf.reduce_sum(labels * tf.log(logistic), axis=1))
        tf.summary.scalar("loss", loss)
    return loss


def create_train_op(loss, global_step_tensor, learning_rate=0.01):
    """
    构建训练对象
    :param loss:
    :param learning_rate:
    :return:
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss=loss, global_step=global_step_tensor)
    return train_op


def create_acc(labels, predictions):
    """
    构建准确率
    :param labels:
    :param predictions:
    :return:
    """
    with tf.variable_scope("acc"):
        # 得到实际的y的标签值
        y_label = tf.argmax(labels, 1)
        # 比较实际值和预测值，然后将比较的结果转换为Float类型(true->1, false->0)；求这个列表/Tensor的均值
        acc = tf.reduce_mean(tf.cast(tf.equal(y_label, predictions), tf.float32))
        tf.summary.scalar("acc", acc)
    return acc


if __name__ == '__main__':
    # 一、执行流图的构建
    # 1.0 定义一个全局变量
    global_step_tensor = tf.train.get_or_create_global_step()
    # 1.1 定义占位符
    input_x = tf.placeholder(dtype=tf.float32, shape=[None, 784], name='input_x')
    input_y = tf.placeholder(dtype=tf.float32, shape=[None, 10], name='input_y')
    # 1.2 网络模型对象构建
    p, y_predict = create_model(input_x)
    # 1.3 损失函数定义构建
    loss = create_loss(input_y, p)
    # 1.4 定义训练对象
    train_op = create_train_op(loss, global_step_tensor)
    # 1.5 模型评估对象
    acc = create_acc(input_y, y_predict)

    # 二、图的训练执行
    # 2.1 数据加载
    mnist = input_data.read_data_sets(train_dir='../datas/mnist', one_hot=True, validation_size=0)

    # 2.2. 获取训练数据
    # 训练数据只负责模型的训练
    train_number = mnist.train.num_examples
    train_images = mnist.train.images
    train_labels = mnist.train.labels
    print("训练数据，特征属性形状:{}, 目标属性形状:{}, 样本数目:{}".format(train_images.shape, train_labels.shape, train_number))

    # c. 运行训练
    with tf.Session() as sess:
        # 可视化相关代码添加
        writer = tf.summary.FileWriter("./model/test06/graph", sess.graph)
        summary_merge_op = tf.summary.merge_all()

        # 添加一个持久化模型的相关对象
        saver = tf.train.Saver()

        # 根据持久化的模型文件是否存在做一个加载的操作
        # ckpt = tf.train.latest_checkpoint('./model/test06')
        # if ckpt is None:
        #     # 表示没有缓冲的文件存在
        #     # 初始化变量
        #     print("初始化相关变量...")
        #     tf.global_variables_initializer().run()
        # else:
        #     # 做模型的恢复
        #     print("加载模型....")
        #     saver.restore(sess, ckpt)
        ckpt_state = tf.train.get_checkpoint_state('./model/test06')
        if ckpt_state and ckpt_state.model_checkpoint_path:
            # 做模型的恢复
            print("加载模型....")
            # 恢复模型参数
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            # 恢复模型的管理对象
            saver.recover_last_checkpoints(checkpoint_paths=ckpt_state.all_model_checkpoint_paths)
        else:
            # 表示没有缓冲的文件存在
            # 初始化变量
            print("初始化相关变量...")
            tf.global_variables_initializer().run()

        # 获取全局步骤对象
        global_step = sess.run(global_step_tensor)
        print(global_step)
        # 迭代进行训练
        for epoch in range(2):
            # 每个批次训练多少数据
            batch_size = 64
            # 总的批次数目
            total_batch = train_number // batch_size
            # 针对当前的Epoch产生一个打乱顺序的下标序列列表
            random_index = np.random.permutation(train_number)
            # 迭代每个批次
            for k in range(total_batch):
                # 获取当前批次的训练数据
                # a. 获取当前批次对应的样本索引
                start_index = k * batch_size
                end_index = int(min(train_number, (k + 1) * batch_size))
                idx = random_index[start_index:end_index]
                # b. 根据索引获取训练数据
                batch_images = train_images[idx]
                batch_labels = train_labels[idx]
                # c. 训练模型
                feed_dict = {
                    input_x: batch_images,
                    input_y: batch_labels
                }
                _, _loss, _acc, _summary = sess.run([train_op, loss, acc, summary_merge_op], feed_dict=feed_dict)
                print("第{}次Epoch中第{}个批次训练后模型\n损失函数值:{}；准确率:{}".format(epoch + 1, k + 1, _loss, _acc))
                writer.add_summary(_summary, global_step=global_step)

                # 进行模型持久化操作
                if k % 100 == 0:
                    saver.save(sess, './model/test06/model.ckpt', global_step=global_step)
                global_step += 1

        # 模型训练完成后，做一个模型保存
        saver.save(sess, './model/test06/model.ckpt', global_step=global_step)

        # 模型的验证预测，效果评估
        test_images = mnist.test.images
        test_labels = mnist.test.labels
        print("测试数据，特征属性形状:{}, 目标属性形状:{}".format(test_images.shape, test_labels.shape))
        _y_predict, _test_acc = sess.run([y_predict, acc], feed_dict={input_x: test_images, input_y: test_labels})
        print("训练数据上的准确率为:{}".format(_test_acc))
