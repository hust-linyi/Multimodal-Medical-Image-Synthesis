import os
from net import *
import numpy as np
import cv2
import data

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
IMAGE_SIZE = 64
BATCH_SIZE = 483
CHANNEL = 1
ADC_DIR = ''
NAME_LIST_PATH_ADC = ''
SAVE_ADC_PATH = ''
SAVE_T2_PATH = ''
results_path = ''
readme = 'temp'
z_dim = 128

z = tf.placeholder(tf.float32, shape=[BATCH_SIZE, z_dim], name='z')


def form_results():
    """
    Forms folders for each run to store the tensorboard files, saved models and the log files.
    :return: three string pointing to tensorboard, saved models and log paths respectively.
    """
    folder_name = "/{0}".format(readme)
    save_adc_path = results_path + folder_name + '/ADC/'
    save_t2_path = results_path + folder_name + '/T2/'
    if not os.path.exists(results_path + folder_name):
        os.mkdir(results_path + folder_name)
        os.mkdir(save_adc_path)
        os.mkdir(save_t2_path)
    return save_adc_path, save_t2_path

def Save_adc():
    fake_adc = Generator_ADC(z)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        save_path = tf.train.latest_checkpoint('./Results/ADC/2018-04-24__22:03:12__stitch/Saved_models/')
        saver.restore(sess, save_path)
        input_z = np.random.randn(BATCH_SIZE, z_dim)
        _fake_adc = sess.run(fake_adc, feed_dict={z: input_z})
        _fake_adc = tf.squeeze(_fake_adc).eval()
        for i in range(BATCH_SIZE):
            cv2.imwrite(SAVE_ADC_PATH + str(i) + '.png', ((_fake_adc[i] + 1) * 255 / 2).astype('int32'))
        print('finishing save ADC image in {}'.format(SAVE_ADC_PATH))

def Save_pair():
    fake_adc = Generator_ADC(z)
    fake_t2 = Generator_T2(fake_adc)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        all_results = os.listdir(results_path)
        all_results.sort()
        save_path = tf.train.latest_checkpoint(results_path + all_results[-1] + '/Saved_models/')
        print(save_path)
        saver.restore(sess, save_path)
        input_z = np.random.randn(BATCH_SIZE, z_dim)
        _fake_adc, _fake_t2 = sess.run([fake_adc, fake_t2], feed_dict={z: input_z})
        _fake_adc = tf.squeeze(_fake_adc).eval()
        _fake_t2 = tf.squeeze(_fake_t2).eval()
        for i in range(BATCH_SIZE):
            cv2.imwrite(SAVE_ADC_PATH + str(i) + '.png', ((_fake_adc[i] + 1) * 255 / 2).astype('int32'))
            cv2.imwrite(SAVE_T2_PATH + str(i) + '.png', ((_fake_t2[i] + 1) * 255 / 2).astype('int32'))
        print('finishing save image pair')

def Save_real_to_fake_pair():
    all_real_data_adc = tf.placeholder(tf.int32, shape=[BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, CHANNEL])
    real_data_adc = 2 * ((tf.cast(all_real_data_adc, tf.float32) / 255.) - .5)

    test_gen_adc = data.load(batch_size=BATCH_SIZE,
                                        data_dir=ADC_DIR,
                                        name_list_path=NAME_LIST_PATH_ADC,
                                        size=(IMAGE_SIZE, IMAGE_SIZE))
    def inf_test_gen_adc():
        while True:
            for (images,) in test_gen_adc():
                yield images

    _data_adc = inf_test_gen_adc().next()

    real_z = Encoder(real_data_adc)
    fake_data_adc = Generator_ADC(real_z)
    fake_data_t2 = Generator_T2(fake_data_adc)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        all_results = os.listdir(results_path)
        all_results.sort()
        save_path = tf.train.latest_checkpoint(results_path + all_results[-1] + '/Saved_models/')
        print(save_path)
        saver.restore(sess, save_path)
        _fake_adc, _fake_t2 = sess.run([fake_data_adc, fake_data_t2], feed_dict={all_real_data_adc: _data_adc})
        _fake_adc = tf.squeeze(_fake_adc).eval()
        _fake_t2 = tf.squeeze(_fake_t2).eval()
        for i in range(BATCH_SIZE):
            cv2.imwrite(SAVE_ADC_PATH + str(i) + '.png', ((_fake_adc[i] + 1) * 255 / 2).astype('int32'))
            cv2.imwrite(SAVE_T2_PATH + str(i) + '.png', ((_fake_t2[i] + 1) * 255 / 2).astype('int32'))
        print('finishing save image pair')

if __name__ == '__main__':
    Save_real_to_fake_pair()
