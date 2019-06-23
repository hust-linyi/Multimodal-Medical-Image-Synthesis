import tensorflow as tf
import tensorflow.contrib.slim as slim

IMAGE_SIZE = 64
z_dim = 128

def Encoder(tensor):
    reuse = len([t for t in tf.global_variables()
                 if t.name.startswith('Encoder')]) > 0
    with tf.variable_scope('Encoder', reuse=reuse):
        tensor = slim.conv2d(tensor, num_outputs=64, kernel_size=[4, 4], stride=2)
        tensor = slim.conv2d(tensor, num_outputs=128, kernel_size=[4, 4], stride=2)
        tensor = slim.flatten(tensor)
        tensor = slim.fully_connected(tensor, num_outputs=1024)
        tensor = slim.fully_connected(tensor, num_outputs=z_dim, activation_fn=None)
    return tensor

def Share_layers(tensor, num_output, stride, scope, is_train=True):
    with tf.variable_scope('sl', reuse=tf.AUTO_REUSE):
        tensor = slim.conv2d_transpose(tensor, num_outputs=num_output, kernel_size=[4, 4], stride=stride,
                                       scope=scope, trainable=is_train)
    return tensor


def Generator_ADC(tensor):
    t_var = [t for t in tf.global_variables()]
    reuse = len([t for t in tf.global_variables()
                 if t.name.startswith('Generator_ADC')]) > 0
    with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(0.)):
        with tf.variable_scope('Generator_ADC', reuse=reuse):
            tensor = slim.fully_connected(tensor, num_outputs=128, scope='fc1')
            tensor = slim.fully_connected(tensor, num_outputs=1024, scope='fc2')
            tensor = slim.fully_connected(tensor, num_outputs=2 * 2 * 512, scope='fc3')
            tensor = tf.reshape(tensor, shape=[-1, 2, 2, 512])
        tensor = Share_layers(tensor, num_output=512, stride=2, scope='dcov1')
        tensor = Share_layers(tensor, num_output=256, stride=2, scope='dcov2')
        with tf.variable_scope('Generator_ADC', reuse=reuse):
            tensor = slim.conv2d_transpose(tensor, num_outputs=128, kernel_size=[4, 4], stride=2, scope='dcov3')
            tensor = tf.depth_to_space(tensor, 2)
            tensor = slim.conv2d_transpose(tensor, num_outputs=1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.tanh, scope='dcov4')
    return tensor


def Discriminator_ADC(tensor):
    reuse = len([t for t in tf.global_variables()
                 if t.name.startswith('Discriminator_ADC')]) > 0
    with tf.variable_scope('Discriminator_ADC', reuse=reuse):
        tensor = slim.conv2d(tensor, num_outputs=64, kernel_size=[4, 4], stride=2)
        tensor = slim.conv2d(tensor, num_outputs=128, kernel_size=[4, 4], stride=2)
        tensor = slim.flatten(tensor)
        tensor = slim.fully_connected(tensor, num_outputs=1024)
        tensor = slim.fully_connected(tensor, num_outputs=128)
        tensor = slim.fully_connected(tensor, num_outputs=1, activation_fn=None)
    return tensor


def Generator_T2(tensor):
    reuse = len([t for t in tf.global_variables()
                 if t.name.startswith('Generator_T2')]) > 0
    with tf.variable_scope('Generator_T2', reuse=reuse):
        layers = []
        c = 64
        # layer_1: [batch, 64, 64, 1] => [batch, 32, 32, 64]
        output = slim.conv2d(tensor, num_outputs=c, kernel_size=[4, 4], stride=2)
        layers.append(output)
        layer_specs = [
            c * 2,  # encoder_2: [batch, 32, 32, c] => [batch, 16, 16, c * 2]
            c * 4,  # encoder_3: [batch, 16, 16, c * 2] => [batch, 8, 8, c * 4]
            c * 8,  # encoder_4: [batch, 8, 8, c * 4] => [batch, 4, 4, c * 8]
            c * 8,  # encoder_5: [batch, 4, 4, c * 8] => [batch, 2, 2, c * 8]
        ]
        for output_channels in layer_specs:
            output = slim.conv2d(layers[-1], num_outputs=output_channels, kernel_size=[4, 4], stride=2)
            layers.append(output)

    # [batch, 2, 2, c * 8] => [batch, 4, 4, c * 8 ]
    output = Share_layers(layers[-1], num_output=512, stride=2, scope='dcov1')
    layers.append(output)
    # [batch, 4, 4, c * 8] => [batch, 8, 8, c * 4]
    output_1 = Share_layers(layers[-1], num_output=256, stride=2, scope='dcov2')

    with tf.variable_scope('Generator_T2', reuse=reuse):
        output_2 = slim.conv2d_transpose(layers[3], num_outputs=256, kernel_size=[4, 4], stride=2, scope='dcov2')
        output = output_1 + output_2
        layers.append(output)
        #[batch, 8, 8, c * 4] => [batch, 16, 16, c *2]
        input = tf.concat([layers[-1], layers[2]], axis=3)
        output = slim.conv2d_transpose(input, num_outputs=128, kernel_size=[4, 4], stride=2, scope='dcov3')
        layers.append(output)
        #[batch, 16, 16, c * 2] = > [batch, 32, 32, c]
        input = tf.concat([layers[-1], layers[1]], axis=3)
        output = slim.conv2d_transpose(input, num_outputs=64, kernel_size=[4, 4], stride=2, scope='dcov4')
        layers.append(output)
        #[batch, 32, 32, c] => [batch, 64, 64, 1]
        input = tf.concat([layers[-1], layers[0]], axis=3)
        output = slim.conv2d_transpose(input, num_outputs=1, kernel_size=[4, 4], stride=2, scope='dcov5', activation_fn=tf.nn.tanh)
        layers.append(output)

    return layers[-1]


def Discriminator_T2(tensor):
    reuse = len([t for t in tf.global_variables()
                 if t.name.startswith('Discriminator_T2')]) > 0
    with tf.variable_scope('Discriminator_T2', reuse=reuse):
        tensor = slim.conv2d(tensor, num_outputs=64, kernel_size=[4, 4], stride=2)
        tensor = slim.conv2d(tensor, num_outputs=128, kernel_size=[4, 4], stride=2)
        tensor = slim.flatten(tensor)
        tensor = slim.fully_connected(tensor, num_outputs=1024)
        tensor = slim.fully_connected(tensor, num_outputs=128)
        tensor = slim.fully_connected(tensor, num_outputs=1, activation_fn=None)
    return tensor

