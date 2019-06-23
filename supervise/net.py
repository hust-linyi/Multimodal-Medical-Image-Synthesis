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

def Generator_ADC(tensor):
    reuse = len([t for t in tf.global_variables()
                 if t.name.startswith('Generator_ADC')]) > 0
    with slim.arg_scope([slim.fully_connected, slim.conv2d_transpose],
            weights_regularizer=slim.l2_regularizer(0.)):
        with tf.variable_scope('Generator_ADC', reuse=reuse):
            tensor = slim.fully_connected(tensor, num_outputs=1024, scope='fc1')
            tensor = slim.fully_connected(tensor, num_outputs=16*16*128, scope = 'fc2')
            tensor = tf.reshape(tensor, shape=[-1, 16, 16, 128])
            tensor = slim.conv2d_transpose(tensor, num_outputs=64, kernel_size=[4, 4], stride=2, scope='dcov1')
            tensor = slim.conv2d_transpose(tensor, num_outputs=1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.tanh, scope='dcov2')
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
            c * 8  # encoder_6: [batch, 2, 2, c * 8] => [batch, 1, 1, c * 8]
        ]
        for output_channels in layer_specs:
            output = slim.conv2d(layers[-1], num_outputs=output_channels, kernel_size=[4, 4], stride=2)
            layers.append(output)
        layer_specs = [
            c * 8,  # decoder_6: [batch, 1, 1, c * 8] => [batch, 2, 2, c * 8 ]
            c * 8,  # decoder_5: [batch, 2, 2, c * 8] => [batch, 4, 4, c * 8 ]
            c * 4,  # decoder_4: [batch, 4, 4, c * 8] => [batch, 8, 8, c * 4]
            c * 2,  # decoder_3: [batch, 8, 8, c * 4] => [batch, 16, 16, c *2]
            c,  # decoder_2: [batch, 16, 16, c * 2] => [batch, 32, 32, c]
        ]
        num_encoder_layers = len(layers)
        for decoder_layer, (output_channels) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            if decoder_layer == 0:
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)
            output = slim.conv2d_transpose(input, num_outputs=output_channels, kernel_size=[4, 4], stride=2)
            layers.append(output)
        # decoder_1: [batch, 32, 32, c] => [batch, 64, 64, 1]
        input = tf.concat([layers[-1], layers[0]], axis=3)
        output = slim.conv2d_transpose(input, num_outputs=1, kernel_size=[4, 4], stride=2, activation_fn=tf.nn.tanh)
        layers.append(output)
    return layers[-1]


def Discriminator_z(tensor):
    reuse = len([t for t in tf.global_variables()
                 if t.name.startswith('Discriminator_z')]) > 0
    with tf.variable_scope('Discriminator_z', reuse=reuse):
        tensor = slim.fully_connected(tensor, num_outputs=128)
        tensor = slim.fully_connected(tensor, num_outputs=1024)
        tensor = slim.fully_connected(tensor, num_outputs=1, activation_fn=None)
    return tensor