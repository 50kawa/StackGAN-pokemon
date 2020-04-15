from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import PIL
import os
import os.path
import codecs
import pickle
import random
import numpy as np
import pandas as pd
from miscc.config import cfg

import torch.utils.data as data
from PIL import Image
import os
import os.path
import six
import string
import sys
import torch
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


class PokemonDataset(data.Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        with open(os.path.join(data_dir, 'pokemondata.pkl'),"rb") as f:
            self.pokemon_dataset = pickle.load(f)

        # おまじない
        self.transform = transform
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        self.target_transform = target_transform

        self.pokemon_dict = self.pokemon_dataset[0]
        self.filenames = list(self.pokemon_dict.keys())
        self.iterator = self.prepair_training_pairs

    def prepair_training_pairs(self, index):
        key = self.filenames[index]
        return self.pokemon_dict[key][0], self.pokemon_dict[key][1], key  # captions

    def __getitem__(self, index):
        return self.iterator(index)

    def __len__(self):
        return len(self.filenames)