#encoding-utf-8
from __future__ import print_function
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.autograd import Variable

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import cloudpickle
import pickle
import sys

sys.path.append('../')

from miscc.config import cfg, cfg_from_file
from trainer import train, evaluate

def parse_args():
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/birds_proGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    parser.add_argument('--model_path', dest='model_path', type=str, default='./')
    parser.add_argument('--char_dict', dest='char_dict', type=str, default='./')
    args = parser.parse_args()
    return args


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    # with open(args.model_path, 'rb') as f:
    #     model = cloudpickle.load(f)
    from model import Seq2Seq, Encoder, Decoder
    encoder = Encoder()
    encoder.apply(weights_init)
    encoder = torch.nn.DataParallel(encoder, device_ids=[int(ix) for ix in cfg.GPU_ID.split(',')])
    decoder = Decoder()
    decoder.apply(weights_init)
    decoder = torch.nn.DataParallel(decoder, device_ids=[int(ix) for ix in cfg.GPU_ID.split(',')])
    model = Seq2Seq(encoder, decoder, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.load_state_dict(torch.load(args.model_path))
    with open(args.char_dict, 'rb') as f:
        data = pickle.load(f)
        char_dict = data[1]
        char_id_dict = data[2]
        type_dict = data[3]
        type_id_dict = data[4]
    model.eval()
    cfg.TRAIN.BATCH_SIZE = 1
    while True:

        # 名前を入力
        char_list = [0] * 6
        while True:
            print('pokemon name:', end='')
            char_input = input()
            # char_input="チョボマキ"
            if len(char_input) > 6:
                print('max input len is 6')
                continue
            f = 0
            for i in range(len(char_input)):
                if char_input[i] not in char_dict:
                    print("invalid char at " + str[i])
                    f = 1
                    break
                char_list[i] = char_dict[char_input[i]]
            if f == 0:
                break
            else:
                char_list = [0] * 6
                continue


        type = [0] * 24

        # タイプを入力
        while True:
            print('type split by 、 :', end='')
            type_input = input().split('、')
            # type_input = "むし".split('、')
            if len(type_input)>2:
                print('max type len is 2')
                continue

            f = 0
            for t in type_input:
                if t not in type_dict:
                    print("invalid type: " + t)
                    f = 1
                    break

                type[type_dict[t]] = 1

            if f == 0:
                break
            else:
                type = [0] * 24
                continue

        # パラメータを入力
        while True:
            print('habcds split by , :', end='')
            habcds = input().split(',')
            # habcds = "50,40,85,40,65,25".split(',')
            if len(habcds)!=6:
                print('please input 6 values')
                continue

            f = 0
            for i in habcds:
                if i.isdecimal():
                    continue
                else:
                    f = 1
                    break
            if f == 1:
                print('some value is not a number')
                continue
            type[18] = int(habcds[0])
            type[19] = int(habcds[1])
            type[20] = int(habcds[2])
            type[21] = int(habcds[3])
            type[22] = int(habcds[4])
            type[23] = int(habcds[5])
            break

        if cfg.CUDA:
            input_char = Variable(torch.t(torch.LongTensor([char_list]))).cuda()
            input_poke = Variable(torch.LongTensor([type]).type(torch.FloatTensor)).cuda()
        else:
            input_char = Variable(torch.t(torch.LongTensor([char_list])))
            input_poke = Variable(torch.LongTensor([type]).type(torch.FloatTensor))
        output_char, output_poke = model(input_char, input_poke, 0) # turn off teacher forcing
        output_char = output_char.tolist()
        output_poke = output_poke.tolist()
        print("pokemon name:", end="")
        for c in output_char:
            i = c[0].index(max(c[0]))
            print(char_id_dict[i], end="")
        print("")
        print("type",end="")
        output_poke_type = output_poke[0][:18]
        output_poke_habcds = output_poke[0][18:]
        max_values = sorted(output_poke_type, reverse=True)[:2]
        max_indexes = [output_poke_type.index(max_values[0]), output_poke_type.index(max_values[1])]
        if max_values[0]/2 < max_values[1]:
            print(type_id_dict[max_indexes[0]] + " " + type_id_dict[max_indexes[1]], end="")
        else:
            print(type_id_dict[max_indexes[0]], end="")
        print("")
        print(str(int(output_poke_habcds[0])) + " ", end="")
        print(str(int(output_poke_habcds[1])) + " ", end="")
        print(str(int(output_poke_habcds[2])) + " ", end="")
        print(str(int(output_poke_habcds[3])) + " ", end="")
        print(str(int(output_poke_habcds[4])) + " ", end="")
        print(str(int(output_poke_habcds[5])) + " ", end="")
        print("")

