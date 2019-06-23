import numpy as np
from net import *
import data
import os
import datetime
import cv2

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
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

gen1_costs, gen2_costs, dc1_costs, dc2_costs = [], [], [], []
for device_index, (device, data_adc, data_t2, data_z) in \
        enumerate(zip(DEVICES, split_real_data_adc, split_real_data_t2, split_z)):
    with tf.device(device):
        real_data_adc = 2 * ((tf.cast(data_adc, tf.float32) / 255.) - .5)
        real_data_t2 = 2 * ((tf.cast(data_t2, tf.float32) / 255.) - .5)

        fake_adc = Generator_ADC(z)
        fake_t2 = Generator_T2(fake_adc)

        #d1_loss
        dc1_real = Discriminator_ADC(real_data_adc)
        dc1_fake = Discriminator_ADC(fake_adc)
        dc1_loss = tf.reduce_mean(dc1_fake) - tf.reduce_mean(dc1_real)

        #d2_loss
        dc2_real = Discriminator_T2(real_data_t2)
        dc2_fake = Discriminator_T2(fake_t2)
        dc2_loss = tf.reduce_mean(dc2_fake) - tf.reduce_mean(dc2_real)

        # gradient penalty
        def gradient_penalty(real, fake, name):
            real = tf.reshape(real, [int(BATCH_SIZE / len(DEVICES)), OUTPUT_DIM])
            fake = tf.reshape(fake, [int(BATCH_SIZE / len(DEVICES)), OUTPUT_DIM])

            alpha = tf.random_uniform(
                shape=[int(BATCH_SIZE / len(DEVICES)), 1],
                minval=0.,
                maxval=1.
            )

            differences = fake - real
            interpolates = real + (alpha * differences)
            interpolates = tf.reshape(interpolates, [-1, IMAGE_SIZE, IMAGE_SIZE, 1])
            if name is 'adc':
                gradients = tf.gradients(Discriminator_ADC(interpolates), [interpolates])[0]
            else:
                gradients = tf.gradients(Discriminator_T2(interpolates), [interpolates])[0]
            slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
            gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
            return gradient_penalty

        gp_adc = gradient_penalty(real_data_adc, fake_adc, 'adc')
        dc1_loss += 10 * gp_adc
        gp_t2 = gradient_penalty(real_data_t2, fake_t2, 't2')
        dc2_loss += 10 * gp_t2

        # gen loss
        gen1_loss = -tf.reduce_mean(dc1_fake)
        gen2_loss = -tf.reduce_mean(dc2_fake)

        tf.summary.image("Input ADC", real_data_adc, max_outputs=5)
        tf.summary.image("Input T2", real_data_t2, max_outputs=5)
        tf.summary.image("Generated ADC", fake_adc, max_outputs=10)
        tf.summary.image("Generated T2", fake_t2, max_outputs=10)

        gen1_costs.append(gen1_loss)
        gen2_costs.append(gen2_loss)
        dc1_costs.append(dc1_loss)
        dc2_costs.append(dc2_loss)

gen1_cost = tf.add_n(gen1_costs) / len(DEVICES)
gen2_cost = tf.add_n(gen2_costs) / len(DEVICES)
dc1_cost = tf.add_n(dc1_costs) / len(DEVICES)
dc2_cost = tf.add_n(dc2_costs) / len(DEVICES)

tf.summary.scalar("D1 Loss", dc1_cost)
tf.summary.scalar("D2 Loss", dc2_cost)
tf.summary.scalar("Gen1 Loss", gen1_cost)
tf.summary.scalar("Gen2 Loss", gen2_cost)
summary_op = tf.summary.merge_all()

t_vars = tf.trainable_variables()
g1_var = [t for t in t_vars if 'Generator_ADC' in t.name]
g2_var = [t for t in t_vars if 'Generator_T2' in t.name]
d1_var = [t for t in t_vars if 'Discriminator_ADC' in t.name]
d2_var = [t for t in t_vars if 'Discriminator_T2' in t.name]


dc1_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                     beta1=0.,
                                     beta2=0.9).minimize(dc1_cost,
                                                         var_list=d1_var, colocate_gradients_with_ops=True)
dc2_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                      beta1=0.,
                                      beta2=0.9).minimize(dc2_cost,
                                                          var_list=d2_var, colocate_gradients_with_ops=True)
gen1_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.,
                                       beta2=0.9).minimize(gen1_cost,
                                                           var_list=g1_var, colocate_gradients_with_ops=True)

gen2_train_op = tf.train.AdamOptimizer(learning_rate=learning_rate,
                                       beta1=0.,
                                       beta2=0.9).minimize(gen2_cost,
                                                           var_list=g2_var, colocate_gradients_with_ops=True)


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
saver = tf.train.Saver(max_to_keep=10)
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
                                        name_list_path=NAME_LIST_PATH_T2,
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

        sess.run(dc1_train_op, feed_dict={all_real_data_adc: _data_adc, z: z_real_dist})
        sess.run(dc2_train_op, feed_dict={all_real_data_t2: _data_t2, z: z_real_dist})

        if i % 3 == 0:
            sess.run(gen1_train_op, feed_dict={z: z_real_dist})
            for j in range(2):
                sess.run(gen2_train_op, feed_dict={z: z_real_dist})

        if i % 50 == 0:
            _dc1_loss, _dc2_loss, _gen1_loss, _gen2_loss, summary = sess.run(
                [dc1_cost, dc2_cost, gen1_cost, gen2_cost, summary_op],
                feed_dict={all_real_data_adc: _data_adc, all_real_data_t2: _data_t2, z: z_real_dist})
            print("iteration:{}".format(i))
            print("D1  Loss={}".format(_dc1_loss))
            print("D2  Loss={}".format(_dc2_loss))
            print("G1  Loss={}".format(_gen1_loss))
            print("G2  Loss={}".format(_gen2_loss))
            writer.add_summary(summary, global_step=i)

#        if i> 19000 and i % 1000 == 0:
#            saver.save(sess, save_path=saved_model_path, global_step=i)
        if i > 10000 and i % 1000 ==0:
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




