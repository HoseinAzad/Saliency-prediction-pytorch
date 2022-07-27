import fnmatch
import os
import os.path
from random import randrange
import numpy as np
import matplotlib.pyplot as plt


def load_data(data_path, split):
    if not os.path.isdir(data_path):
        print('Dataset path does not exist ')
        quit()

    images = get_all_files(data_path + '/Stimuli', extension='jpg')
    maps = get_all_files(data_path + '/FIXATIONMAPS', extension='jpg')
    # It's necessary to sort images and maps path-list by their names (here, the name play the role of id)
    images.sort()
    maps.sort()
    return split_data(np.array(images), np.array(maps), split)


def get_all_files(path, extension):
    class_id = 0
    files = []
    for root, dirNames, fileNames in os.walk(path):
        if not ('Output' in root):
            # In the following, we specify each folder name with an id (here, with the "class_id" variable)
            # Finally, we will use this id to split data to train and validation set with equal sampling in each class
            if not (len(fileNames) == 0):
                class_id += 1
                for fileName in fnmatch.filter(fileNames, '*.' + extension):
                    files.append([class_id, os.path.join(root, fileName)])

    return files


# Split data to train and validation set with maintaining the frequency of samples in different dataset classes
def split_data(images, maps, split):
    img_train, map_train, img_val, map_val = [], [], [], []
    classes = np.unique(images[:, 0])

    for c in classes:
        img_sub_val, map_sub_val = [], []
        # Select all samples in class c
        img_sub_train = images[images[:, 0] == c, 1].tolist()
        map_sub_train = maps[maps[:, 0] == c, 1].tolist()
        sub_trian_size = len(img_sub_train)
        # until condition satisfies -> pop sample from sub_train and push it on sub_val
        while len(img_sub_train) > (split * sub_trian_size):
            index = randrange(len(img_sub_train))
            img_sub_val.append(img_sub_train.pop(index))
            map_sub_val.append(map_sub_train.pop(index))

        img_train = img_train + img_sub_train
        img_val = img_val + img_sub_val
        map_train = map_train + map_sub_train
        map_val = map_val + map_sub_val

    return np.array(img_train), np.array(img_val), np.array(map_train), np.array(map_val)


def display_map(img, op, map):
    f, ax = plt.subplots(1, 3, figsize=(14, 10), dpi=100)
    for a in ax:
        a.set_axis_off()

    ax[0].set_title('Raw Image', size=10)
    ax[0].imshow(img)
    ax[1].set_title('Ground Truth Saliency Map', size=10)
    ax[1].imshow(map)
    ax[2].set_title('Network Output', size=10)
    ax[2].imshow(op)
    plt.show()


def plot_loss(loss_list, title):
    plt.plot(loss_list[:, 0], '-o', label="train-loss")
    plt.plot(loss_list[:, 1], '-o', label="validation-loss")
    plt.title("Loss Curves - " + title, fontsize=12)
    plt.ylabel("Loss", fontsize=10)
    plt.xlabel("Epoch", fontsize=10)
    plt.legend(prop={'size': 10})
    plt.show()
