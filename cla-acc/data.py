import numpy as np
import cv2


def make_generator(n_files, batch_size, name_list, label_list, data_dir_1, data_dir_2, size=(64, 64)):
    epoch_count = [0]
    n_iter = int(n_files / batch_size)
    n_rest = int(n_files % batch_size)
    def get_epoch():
        images = np.zeros((batch_size, 64, 64, 1), dtype='int32')
        labels = np.zeros((batch_size,), dtype='int32')
        namelist = np.chararray((batch_size,), itemsize=11)
        files = list(range(n_files))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n in range(n_iter):
            indice = files[n*batch_size:(n+1)*batch_size]
            for i, j in enumerate(indice):
                images[i] = read_image_by_size(data_dir_1, data_dir_2, name_list[j], size=size)
                labels[i] = label_list[j]
                namelist[i] = name_list[j]
            yield (images, labels, namelist)
        if n_rest is not 0:
            indice = files[n_iter*batch_size:]
            for i, j in enumerate(indice):
                images[i] = read_image_by_size(data_dir_1, data_dir_2, name_list[j], size=size)

                labels[i] = label_list[j]
                namelist[i] = name_list[j]
            yield (images, labels, namelist)
            indice = files[:batch_size - n_rest]
            for i, j in enumerate(indice):
                images[i] = read_image_by_size(data_dir_1, data_dir_2, name_list[j], size=size)

                labels[-i-1] = label_list[j]
                namelist[-i-1] = name_list[j]
            yield (images, labels, namelist)
    return get_epoch

def load(batch_size, data_dir_1, data_dir_2='', name_list_path=None, size=(64, 64)):
    name_list = []
    label_list = []
    for line in open(name_list_path):
        line = line.strip('\n')
        line = line.strip('\r')
        name, label = line.split(' ')
        name_list.append(name)
        label_list.append(label)
    return make_generator(len(name_list), batch_size, name_list, label_list, data_dir_1, data_dir_2, size=size)

def read_image_by_size(data_dir_1, data_dir_2, name, size=(64, 64), grayscale=True):
    path_1 = data_dir_1 + ''.join(name)
    path_2 = data_dir_2 + ''.join(name)
    image = cv2.imread(path_1)
    if image is None:
        image = cv2.imread(path_2)
        if image is None:
            print(path_1)
    if image.shape[:2] != size:
        image = cv2.resize(image, size)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, :, np.newaxis]
        return image
    else:
        return image.transpose(2, 0, 1)
