import json
import os.path as osp


import torch
from torch.utils.data import Dataset


class ImageNet():

    def __init__(self, path):
        self.path = path
        self.keep_ratio = 1.0

    def get_subset(self, wnid):
        path = osp.join(self.path, wnid +'.json')
        return ImageNetSubset(path, wnid, keep_ratio=self.keep_ratio)

    def set_keep_ratio(self, r):
        self.keep_ratio = r


class ImageNetSubset(Dataset):

    def __init__(self, path, wnid, keep_ratio=1.0):
        self.wnid = wnid

        data = []
        images = json.load(open(path, 'r'))
        for image in images:
            if(image) == None:
                images.remove(image)
            else:
                data.append(torch.tensor(image))

        if images != []:
            self.data = torch.stack(data)
        else:
            self.data = []

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.wnid
