import cv2
import numpy as np
import torch
import torch.utils.data as data


class Dataset(data.Dataset):

    def __init__(self, img_list, map_list, input_width, input_height, output_width, output_height, transform=None):
        # images (model input) and ground truth maps
        self.img_list = img_list
        self.map_list = map_list
        # model input and output size
        self.output_height = output_height
        self.output_width = output_width
        self.input_height = input_height
        self.input_width = input_width

    def __getitem__(self, item):
        img_path = self.img_list[item]
        map_path = self.map_list[item]

        img = cv2.imread(img_path)
        # Create a single channel image (cv2 produces a three-channel image by default)
        map = cv2.cvtColor(cv2.imread(map_path), cv2.COLOR_BGR2GRAY)
        # Resize
        img = cv2.resize(img, (self.input_width, self.input_height), interpolation=cv2.INTER_AREA).astype(np.float32)
        map = cv2.resize(map, (self.output_width, self.output_height), interpolation=cv2.INTER_AREA).astype(np.float32)
        # Normalize data, similar to that described in the paper (mean subtraction and scaling )
        img -= np.array((100, 110, 118))
        img /= (255 / 2)
        map -= np.array(27)
        map /= (255 / 2)
        # Convert numpy to tensor
        img = torch.FloatTensor(img)
        map = torch.FloatTensor(map)
        # Add a dimension to the tensor as the number of channels ( ch=1 )
        map = map.unsqueeze(2)

        return img.permute(2, 0, 1), map.permute(2, 0, 1)

    def __len__(self):
        return len(self.img_list)
