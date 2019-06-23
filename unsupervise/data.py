import numpy as np
import cv2


def make_generator(n_files, batch_size, name_list, data_dir, size=(64, 64)):
    epoch_count = [0]
    n_iter = int(n_files / batch_size)
    n_rest = int(n_files % batch_size)
    def get_epoch():
        images = np.zeros((batch_size, size[0], size[1], 1), dtype='int32')
        files = list(range(n_files))
        random_state = np.random.RandomState(epoch_count[0])
        random_state.shuffle(files)
        epoch_count[0] += 1
        for n in range(n_iter):
            indice = files[n*batch_size:(n+1)*batch_size]
            for i, j in enumerate(indice):
                images[i] = read_image_by_size(data_dir+''.join(name_list[j]), size=size)
            yield (images,)
        if n_rest is not 0:
            indice = files[n_iter*batch_size:]
            for i, j in enumerate(indice):
                images[i] = read_image_by_size(data_dir+''.join(name_list[j]), size=size)
            indice = files[:batch_size - n_rest]
            for i, j in enumerate(indice):
                images[-i-1] = read_image_by_size(data_dir+''.join(name_list[j]), size=size)
            yield (images,)
    return get_epoch

def load(batch_size, data_dir, name_list_path, size=(64, 64)):
    name_list = []
    for line in open(name_list_path):
        line = line.strip('\n')
        line = line.strip('\r')
        name = line.split(' ')
        name_list.append(name)
    return make_generator(len(name_list), batch_size, name_list, data_dir, size=size)

def read_image_by_size(path, size=(64, 64), grayscale=True):
    image = cv2.imread(path)
    if image is None:
        print(path)
    if image.shape[:2] != size:
        image = cv2.resize(image, size)
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = image[:, :, np.newaxis]
        return image
    else:
        return image.transpose(2, 0, 1)
