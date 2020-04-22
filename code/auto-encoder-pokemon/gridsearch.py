from __future__ import print_function
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time
import cloudpickle

import sys
sys.path.append('../')

from miscc.config import cfg, cfg_from_file
from trainer import train, evaluate

def parse_args():
    tp = lambda x: list(map(float, x.split(',')))
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/birds_proGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--model_dir', dest='model_dir', type=str, default='./')
    parser.add_argument('--char_grads', dest='char_grads', type=tp, default='1')
    parser.add_argument('--type_grads', dest='type_grads', type=tp, default='1')
    parser.add_argument('--habcds_grads', dest='habcds_grads', type=tp, default='1')
    parser.add_argument('--learning_rates', dest='learning_rates', type=tp, default='0.001')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    from datasets import PokemonDataset
    dataset = PokemonDataset(args.data_dir, transform=transforms.ToTensor())

    # shuffleしてから分割してくれる.
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [len(dataset) - len(dataset)//4, len(dataset)//4])

    grid_search_result='lr,char,type,habcds,char_loss,type_loss,habcds_loss\n'
    for learning_rate in args.learning_rates:
        cfg.TRAIN.LEARNING_RATE = learning_rate
        for char_grad in args.char_grads:
            cfg.TRAIN.LOSS_GRAD.POKEMON_CHAR = char_grad
            for type_grad in args.type_grads:
                cfg.TRAIN.LOSS_GRAD.POKEMON_TYPE = type_grad
                for habcds_grad in args.habcds_grads:
                    cfg.TRAIN.LOSS_GRAD.POKEMON_HABCDS = habcds_grad

                    from model import Seq2Seq, Encoder, Decoder
                    encoder = Encoder()
                    encoder.apply(weights_init)
                    encoder = torch.nn.DataParallel(encoder, device_ids=[int(ix) for ix in cfg.GPU_ID.split(',')])
                    decoder = Decoder()
                    decoder.apply(weights_init)
                    decoder = torch.nn.DataParallel(decoder, device_ids=[int(ix) for ix in cfg.GPU_ID.split(',')])
                    model = Seq2Seq(encoder, decoder, device)
                    optimizer = optim.Adam(model.parameters(), lr=cfg.TRAIN.LEARNING_RATE)
                    criterion = nn.CrossEntropyLoss()
                    emb_criterion = nn.MSELoss()
                    clip = 1

                    best_valid_loss = float('inf')
                    best_epoch = 0
                    best_valid_char_loss = 0
                    best_valid_type_loss = 0
                    best_valid_habcds_loss = 0

                    train_dataloader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                        drop_last=True, shuffle=True, num_workers=int(cfg.WORKERS))
                    val_dataloader = torch.utils.data.DataLoader(
                        val_dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
                        drop_last=True, shuffle=False, num_workers=int(cfg.WORKERS))

                    early_stop = 0
                    for epoch_idx in range(cfg.TRAIN.MAX_EPOCH):
                        train_char_loss, train_poke_type_loss, train_poke_habcds_loss = train(model, train_dataloader, optimizer, criterion, emb_criterion, clip)
                        valid_char_loss, valid_poke_type_loss, valid_poke_habcds_loss = evaluate(model, val_dataloader, criterion, emb_criterion)
                        train_loss = train_char_loss + train_poke_type_loss + train_poke_habcds_loss
                        valid_loss =  valid_char_loss + valid_poke_type_loss + valid_poke_habcds_loss
                        if valid_loss < best_valid_loss:
                            best_valid_loss = valid_loss
                            best_valid_char_loss = valid_char_loss
                            best_valid_type_loss = valid_poke_type_loss
                            best_valid_habcds_loss = valid_poke_habcds_loss
                            best_model = model
                            best_epoch = epoch_idx
                            early_stop = 0
                        else:
                            early_stop += 1
                            if early_stop >= cfg.TRAIN.EARLY_STOP:
                                break

                    # torch.save(
                    #     best_model.state_dict(),
                    #     '%s/autoencoder_%d_%f_%f_%f_%f.pth' % (args.model_dir, best_epoch, learning_rate, char_grad, type_grad, habcds_grad))
                    grid_search_result +='%f,%f,%f,%f,%f,%f,%f\n' % (learning_rate,char_grad,type_grad,habcds_grad,best_valid_char_loss,best_valid_type_loss,best_valid_habcds_loss)

    with open("result.csv","w") as f:
        f.write(grid_search_result)