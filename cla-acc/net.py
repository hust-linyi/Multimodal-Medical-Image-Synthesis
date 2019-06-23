import tensorflow as tf
import tensorflow.contrib.slim as slim
def lenet(input_adc, input_t2, num_class=2, prediction_fn=slim.softmax):

    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(0.005),
                        reuse=tf.AUTO_REUSE):
        #adc
        output_adc = slim.conv2d(input_adc, 16, [3, 3], scope='conv1')
        output_adc = slim.max_pool2d(output_adc, [2, 2], 2, scope='pool1')

        output_adc = slim.conv2d(output_adc, 64, [3, 3], scope='conv2')
        output_adc = slim.max_pool2d(output_adc, [2, 2], 2, scope='pool2')

        output_adc = slim.conv2d(output_adc, 256, [3, 3], scope='conv3')
        output_adc = slim.max_pool2d(output_adc, [2, 2], 2, scope='pool3')

        output_adc = slim.conv2d(output_adc, 512, [3, 3], scope='conv4')
        output_adc = slim.max_pool2d(output_adc, [2, 2], 2, scope='pool4')

        output_adc = slim.conv2d(output_adc, 1024, [3, 3], scope='conv5')
        output_adc = slim.max_pool2d(output_adc, [2, 2], 2, scope='pool5')

        #t2
        output_t2 = slim.conv2d(input_t2, 16, [3, 3], scope='conv2_1')
        output_t2 = slim.max_pool2d(output_t2, [2, 2], 2, scope='pool2_1')

        output_t2 = slim.conv2d(output_t2, 64, [3, 3], scope='conv2_2')
        output_t2 = slim.max_pool2d(output_t2, [2, 2], 2, scope='pool2_2')

        output_t2 = slim.conv2d(output_t2, 256, [3, 3], scope='conv2_3')
        output_t2 = slim.max_pool2d(output_t2, [2, 2], 2, scope='pool2_3')

        output_t2 = slim.conv2d(output_t2, 512, [3, 3], scope='conv2_4')
        output_t2 = slim.max_pool2d(output_t2, [2, 2], 2, scope='pool2_4')

        output_t2 = slim.conv2d(output_t2, 1024, [3, 3], scope='conv2_5')
        output_t2 = slim.max_pool2d(output_t2, [2, 2], 2, scope='pool2_5')

        #output
        output = tf.concat([output_adc, output_t2], axis=3)
        output = tf.layers.flatten(output)
        output = slim.fully_connected(output, 1024, scope='fc1')
        output = slim.fully_connected(output, 64, scope='fc2')
        logits = slim.fully_connected(output, num_class, activation_fn=None, scope='fnc4')

    return logits
