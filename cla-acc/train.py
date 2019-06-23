from __future__ import print_function
import tensorflow as tf
import net
import lib
import data
import os
import logging
import heapq
import numpy as np
import argparse

os.environ['CUDA_VISIBLE_DEVICES'] = '1'
parser = argparse.ArgumentParser(description='train pairs')
parser.add_argument('--task', '-t', metavar='TASK', default='real', help='100,500,1000 or 1500')

gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 100
LEARNING_RATE_DECAY = 0.99
LEARNING_RATE_BASE = 0.005
MAX_TRAIN_STEP = 10000
IMAGE_SIZE = 64

args = parser.parse_args()
TASK = args.task

TRAIN_IMAGE_LIST_PATH = ''
DATA_DIR_ADC_POSITIVE = ''
DATA_DIR_ADC_NEGATIVE = ''
DATA_DIR_T2_POSITIVE = ''
DATA_DIR_T2_NEGATIVE = ''
LOGDIR = ''

TEST_IMAGE_LIST_PATH = ''
DATA_DIR_ADC_TEST = ''
DATA_DIR_T2_TEST = ''
N_GPUS = 1
DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
SAVE_LOG = False

lib.print_model_settings(locals().copy())

input_adc = tf.placeholder(dtype='float32', shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1))
input_t2 = tf.placeholder(dtype='float32', shape=(None, IMAGE_SIZE, IMAGE_SIZE, 1))
input_label = tf.placeholder(dtype='int32', shape=(None))
image_adc = 2*((tf.cast(input_adc, tf.float32)/255.)-.5)
image_t2 = 2*((tf.cast(input_t2, tf.float32)/255.)-.5)
if tf.__version__.startswith('1.'):
    split_image_adc = tf.split(image_adc, len(DEVICES))
    split_image_t2 = tf.split(image_t2, len(DEVICES))
    split_label = tf.split(input_label, len(DEVICES))
else:
    split_image_adc = tf.split(0, len(DEVICES), image_adc)
    split_image_t2 = tf.split(0, len(DEVICES), image_t2)
    split_label = tf.split(0, len(DEVICES), input_label)
train_costs, accuracies = [], []

classifer = net.lenet

for device_index, (device, data_adc, data_t2,
                   label_conv) in enumerate(zip(DEVICES, split_image_adc,split_image_t2, split_label)):
    with tf.device(device):
        prediction = classifer(data_adc, data_t2)
        correct_prediction = tf.equal(tf.cast(tf.argmax(prediction, 1), tf.int32), label_conv)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        classification_loss = \
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=label_conv))
        train_costs.append(classification_loss)
        accuracies.append(accuracy)

train_cost = tf.add_n(train_costs) / len(DEVICES)
regularization_loss = \
    tf.add_n(tf.losses.get_regularization_losses())

total_loss = train_cost + regularization_loss
total_accuracy = tf.add_n(accuracies) / len(DEVICES)

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, 30,
                                           LEARNING_RATE_DECAY)
train_op = \
        tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(total_loss,
                var_list=tf.trainable_variables(), colocate_gradients_with_ops=True, global_step=global_step)

train_merge = [tf.summary.scalar('train loss/' + 'classification_loss', train_cost),
               tf.summary.scalar('train loss/' + 'regularization_loss', regularization_loss),
               tf.summary.scalar('train loss/' + 'total_loss', total_loss)]
test_merge = [tf.summary.scalar('test accuracy/' + 'accuracy', total_accuracy)]
train_merge_op = tf.summary.merge(train_merge)
test_merge_op = tf.summary.merge(test_merge)

# Dataset iterator
train_gen_adc = data.load(TRAIN_BATCH_SIZE,
                                     data_dir_1=DATA_DIR_ADC_POSITIVE,
                                     data_dir_2=DATA_DIR_ADC_NEGATIVE,
                                     name_list_path=TRAIN_IMAGE_LIST_PATH,
                                     size=(IMAGE_SIZE, IMAGE_SIZE))
train_gen_t2 = data.load(TRAIN_BATCH_SIZE,
                                     data_dir_1=DATA_DIR_T2_POSITIVE,
                                     data_dir_2=DATA_DIR_T2_NEGATIVE,
                                     name_list_path=TRAIN_IMAGE_LIST_PATH,
                                     size=(IMAGE_SIZE, IMAGE_SIZE))
test_gen_adc = data.load(TEST_BATCH_SIZE,
                                data_dir_1=DATA_DIR_ADC_TEST,
                                name_list_path=TEST_IMAGE_LIST_PATH,
                                size=(IMAGE_SIZE, IMAGE_SIZE))
test_gen_t2 = data.load(TEST_BATCH_SIZE,
                                data_dir_1=DATA_DIR_T2_TEST,
                                name_list_path=TEST_IMAGE_LIST_PATH,
                                size=(IMAGE_SIZE, IMAGE_SIZE))
def inf_train_gen_adc():
    while True:
        for (images, labels,_) in train_gen_adc():
            yield images, labels

def inf_train_gen_t2():
    while True:
        for (images, labels,_) in train_gen_t2():
            yield images, labels

def inf_test_gen_adc():
    while True:
        for (images, labels, namelist) in test_gen_adc():
            yield images, labels, namelist

def inf_test_gen_t2():
    while True:
        for (images, labels,_) in test_gen_t2():
            yield images, labels

train_data_adc = inf_train_gen_adc()
train_data_t2 = inf_train_gen_t2()
test_data_adc = inf_test_gen_adc()
test_data_t2 = inf_test_gen_t2()

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=10)

with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(init)
    writer = tf.summary.FileWriter(LOGDIR, sess.graph)
    temp_accuracy = 0
    all_accuracy = []
    for i in range(MAX_TRAIN_STEP):
        if i > MAX_TRAIN_STEP-2:
            top_10 = heapq.nlargest(10, all_accuracy)
            print(top_10)
            average_accuracy = np.mean(top_10)
            std_accuracy = np.std(top_10)
            print('The average of top 10 accuracy is {}'.format(average_accuracy))
            print('The standard deviation of top 20 accuracy is {}'.format(std_accuracy))
            print('The top accuracy is {}'.format(temp_accuracy))
            print(TASK)

            with open('./results_diff_num.txt', 'a') as f:
                f.write('********* train pairs ************\n')
                f.write(TASK)
                f.write('\n')
                f.write('mean acc: {}'.format(average_accuracy))
                f.write('\n')
                f.write('std acc: {}'.format(std_accuracy))
                f.write('\n')

        images_adc, labels_adc = train_data_adc.next()
        images_t2, _ = train_data_t2.next()
        _, step, result = sess.run([train_op, global_step, train_merge_op],
                                   feed_dict={input_adc: images_adc, input_t2: images_t2,
                                              input_label: labels_adc})
        if i % 10 is 0 and SAVE_LOG:
            writer.add_summary(result, step)

        if i % 20 is 0:
            images_adc, labels_adc, namelist_adc = test_data_adc.next()
            images_t2, _ = test_data_t2.next()
            _correct_prediction, accuracy_value, step, result = sess.run([correct_prediction, total_accuracy, global_step, test_merge_op],
                                              feed_dict={input_adc: images_adc, input_t2: images_t2,
                                                         input_label: labels_adc})
            if accuracy_value > 0.9:
               with open(LOGDIR + '/log.txt', 'a') as log:
                   log.write('interation:{}\n'.format(i))
                   log.write('name:{},   label:{}'.format(namelist_adc[i], labels_adc[i]))

            all_accuracy.append(accuracy_value)
            if SAVE_LOG:
              writer.add_summary(result, step)
            if accuracy_value > 0.8:
                if accuracy_value > temp_accuracy:
                    temp_accuracy = accuracy_value
                print('step{},accuracy:{}'.format(step, accuracy_value))
            else:
                print('step:{}'.format(step))
                saver.save(sess, os.path.join(LOGDIR, 'real-classfication'), global_step=global_step)

