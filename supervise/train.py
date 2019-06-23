import numpy as np
from net import *
import data
import os
import datetime
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
IMAGE_SIZE = 64
ADC_DIR = ''
T2_DIR = ''
NAME_LIST_PATH_ADC = ''
NAME_LIST_PATH_T2 = ''
BATCH_SIZE = 32
CHANNEL = 1
OUTPUT_DIM = 64 * 64 * CHANNEL
ITERS = 40000  # How many iterations to train for
N_GPUS = 1  # Number of GPUs
z_dim = 128
learning_rate = 1e-4
beta1 = 0.9
results_path = ''
SAVE_IMAGE_PATH = ''
USE_CPU = False

if USE_CPU:
    DEVICES = ['/cpu:{}'.format(i) for i in range(N_GPUS)]
else:
    DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]

all_real_data_adc = tf.placeholder(tf.int32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL])
all_real_data_t2 = tf.placeholder(tf.int32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL])
z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')

if tf.__version__.startswith('1.'):
    split_real_data_adc = tf.split(all_real_data_adc, len(DEVICES))
    split_real_data_t2 = tf.split(all_real_data_t2, len(DEVICES))
    split_z = tf.split(z, len(DEVICES))
else:
    split_real_data_adc = tf.split(0, len(DEVICES), all_real_data_adc)
    split_real_data_t2 = tf.split(0, len(DEVICES), all_real_data_t2)
    split_z = tf.split(0, len(DEVICES), z)

gen_costs, dc_costs, adc_l1_costs, t2_l1_costs = [], [], [], []
for device_index, (device, data_adc, data_t2, data_z) in \
        enumerate(zip(DEVICES, split_real_data_adc, split_real_data_t2, split_z)):
    with tf.device(device):
        real_data_adc = 2 * ((tf.cast(data_adc, tf.float32) / 255.) - .5)
        real_data_t2 = 2 * ((tf.cast(data_t2, tf.float32) / 255.) - .5)

        real_z = Encoder(real_data_adc)
        fake_data_adc = Generator_ADC(real_z)
        fake_data_t2 = Generator_T2(fake_data_adc)

        fake_adc = Generator_ADC(z)
        fake_t2 = Generator_T2(fake_adc)

        dc_real = Discriminator_z(z)
        dc_fake = Discriminator_z(real_z)

        #l1 loss
        adc_l1_loss = tf.reduce_mean(tf.abs(real_data_adc - fake_data_adc))
        t2_l1_loss = tf.reduce_mean(tf.abs(real_data_t2 - fake_data_t2))

        # discriminator loss
        dc_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dc_real), logits=dc_real))
        dc_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(dc_fake), logits=dc_fake))
        dc_loss = dc_loss_fake + dc_loss_real

        # generator loss
        gen_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(dc_fake), logits=dc_fake))

        tf.summary.image("Input ADC", real_data_adc, max_outputs=5)
        tf.summary.image("Input T2", real_data_t2, max_outputs=5)
        tf.summary.image("Output ADC", fake_data_adc, max_outputs=5)
        tf.summary.image("Output T2", fake_data_t2, max_outputs=5)
        tf.summary.image("Generated ADC", fake_adc, max_outputs=10)
        tf.summary.image("Generated T2", fake_t2, max_outputs=10)

        gen_costs.append(gen_loss)
        dc_costs.append(dc_loss)
        adc_l1_costs.append(adc_l1_loss)
        t2_l1_costs.append(t2_l1_loss)

gen_cost = tf.add_n(gen_costs) / len(DEVICES)
dc_cost = tf.add_n(dc_costs) / len(DEVICES)
adc_l1_cost = tf.add_n(adc_l1_costs) / len(DEVICES)
t2_l1_cost = tf.add_n(t2_l1_costs) / len(DEVICES)

tf.summary.scalar("D Loss", dc_cost)
tf.summary.scalar("Gen Loss", gen_cost)
tf.summary.scalar("adc l1 Loss", adc_l1_cost)
tf.summary.scalar("t2 l1 Loss", adc_l1_cost)
summary_op = tf.summary.merge_all()

t_vars = tf.trainable_variables()
e_var = [t for t in t_vars if 'Encoder' in t.name]
g1_var = [t for t in t_vars if 'Generator_ADC' in t.name]
g2_var = [t for t in t_vars if 'Generator_T2' in t.name]
d_var = [t for t in t_vars if 'Discriminator_z' in t.name]


dc_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                     beta1=0.,
                                     beta2=0.9).minimize(dc_cost,
                                                         var_list=d_var, colocate_gradients_with_ops=True)
gen1_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.,
                                       beta2=0.9).minimize(adc_l1_cost,
                                                           var_list=g1_var + e_var, colocate_gradients_with_ops=True)

gen2_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.,
                                       beta2=0.9).minimize(t2_l1_cost,
                                                           var_list=g2_var, colocate_gradients_with_ops=True)

encoder_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.,
                                       beta2=0.9).minimize(gen_cost,
                                                           var_list=e_var, colocate_gradients_with_ops=True)


def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}".format(datetime.datetime.now().strftime('%Y-%m-%d__%H:%M'))
    tensorboard_path = results_path + folder_name + '/Tensorboard'
    saved_model_path = results_path + folder_name + '/Saved_models/'
    log_path = results_path + folder_name + '/log'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(tensorboard_path)
        os.mkdir(saved_model_path)
        os.mkdir(log_path)
    return tensorboard_path, saved_model_path, log_path


# creating session
saver = tf.train.Saver(max_to_keep=1)
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    # Dataset iterator
    train_gen_adc = data.load(batch_size=BATCH_SIZE,
                                         data_dir=ADC_DIR,
                                         name_list_path=NAME_LIST_PATH_ADC,
                                         size=(IMAGE_SIZE, IMAGE_SIZE))

    def inf_train_gen_adc():
        while True:
            for (images,) in train_gen_adc():
                yield images

    train_gen_t2 = data.load(batch_size=BATCH_SIZE,
                                        data_dir=T2_DIR,
                                        name_list_path=NAME_LIST_PATH_ADC,
                                        size=(IMAGE_SIZE, IMAGE_SIZE))

    def inf_train_gen_t2():
        while True:
            for (images,) in train_gen_t2():
                yield images

    # Train loop

    gen_adc = inf_train_gen_adc()
    gen_t2 = inf_train_gen_t2()
    tensorboard_path, saved_model_path, log_path = form_results()
    writer = tf.summary.FileWriter(logdir=tensorboard_path, graph=sess.graph)
    sess.run(tf.global_variables_initializer())
    for i in range(ITERS):
        z_real_dist = np.random.randn(BATCH_SIZE, z_dim)
        _data_adc = gen_adc.next()
        _data_t2 = gen_t2.next()

        sess.run(dc_train_op, feed_dict={all_real_data_adc: _data_adc, z: z_real_dist})
        sess.run(gen1_train_op, feed_dict={all_real_data_adc: _data_adc})
        sess.run(gen2_train_op, feed_dict={all_real_data_adc: _data_adc, all_real_data_t2: _data_t2})
        sess.run(encoder_train_op, feed_dict={all_real_data_adc: _data_adc, z: z_real_dist})

        if i % 50 == 0:
            _dc_loss, _gen_loss, _adc_l1_loss,_t2_l1_loss, summary = sess.run(
                [dc_cost, gen_cost, adc_l1_cost, t2_l1_cost, summary_op],
                feed_dict={all_real_data_adc: _data_adc, all_real_data_t2: _data_t2, z: z_real_dist})
            print("iteration:{}".format(i))
            print("D  Loss={}".format(_dc_loss))
            print("G  Loss={}".format(_gen_loss))
            print("l1 adc Loss={}".format(_adc_l1_loss))
            print("l1 t2 Loss={}".format(_t2_l1_loss))

            writer.add_summary(summary, global_step=i)

        if i> 10000 and i % 1000 == 0:
            saver.save(sess, save_path=saved_model_path, global_step=i)
        if i > 29000 and i % 1000 ==0:
            generated_z = np.random.randn(500, z_dim)
            _fake_adc, _fake_t2 = sess.run([fake_adc, fake_t2], feed_dict={z: generated_z})
            _fake_adc = tf.squeeze(_fake_adc).eval()
            _fake_t2 = tf.squeeze(_fake_t2).eval()
            SAVE_ADC_PATH = SAVE_IMAGE_PATH + str(i) + '/adc/'
            SAVE_T2_PATH = SAVE_IMAGE_PATH + str(i) + '/t2/'
            if not os.path.exists(SAVE_IMAGE_PATH + str(i)):
               os.mkdir(SAVE_IMAGE_PATH + str(i))
               os.mkdir(SAVE_ADC_PATH)
               os.mkdir(SAVE_T2_PATH)
            for j in range(500):
                cv2.imwrite(SAVE_ADC_PATH + str(j) + '.png', ((_fake_adc[j] + 1) * 255 / 2).astype('int32'))
                cv2.imwrite(SAVE_T2_PATH + str(j) + '.png', ((_fake_t2[j] + 1) * 255 / 2).astype('int32'))
            print('finishing save image pair')




