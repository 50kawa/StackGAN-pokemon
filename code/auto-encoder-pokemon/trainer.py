from __future__ import print_function
from six.moves import range

import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
import os
import time
from PIL import Image, ImageFont, ImageDraw
from copy import deepcopy

from miscc.config import cfg
from miscc.utils import mkdir_p

from torch.utils.tensorboard import summary
from torch.utils.tensorboard import FileWriter


def train(model, data_loader, optimizer, criterion, emb_criterion, clip):
    model.train()

    epoch_char_loss = 0
    epoch_poke_loss = 0

    for i, batch in enumerate(data_loader):
        if cfg.CUDA:
            input_char = Variable(torch.stack(batch[0])).cuda()
            input_poke = Variable(torch.t(torch.stack(batch[1]).type(torch.FloatTensor))).cuda()
        else:
            input_char = Variable(torch.stack(batch[0]))
            input_poke = Variable(torch.t(torch.stack(batch[1]).type(torch.FloatTensor)))

        optimizer.zero_grad()

        output_char, output_poke = model(input_char, input_poke)
        # trg = [trg len, batch size]
        # output = [trg len, batch size, output dim]

        output_char_dim = output_char.shape[-1]

        output_char = output_char.view(-1, output_char_dim)
        input_char = input_char.view(-1)

        # trg = [(trg len - 1) * batch size]
        # output = [(trg len - 1) * batch size, output dim]

        loss = criterion(output_char, input_char)

        loss_poke = emb_criterion(output_poke, input_poke)
        all_loss = loss + loss_poke/100
        all_loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_char_loss += loss.item()
        epoch_poke_loss += loss_poke.item()

    return epoch_char_loss / len(data_loader), epoch_poke_loss / len(data_loader)


def evaluate(model, data_loader, criterion, emb_criterion):
    model.eval()

    epoch_char_loss = 0
    epoch_poke_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if cfg.CUDA:
                input_char = Variable(torch.stack(batch[0])).cuda()
                input_poke = Variable(torch.t(torch.stack(batch[1]).type(torch.FloatTensor))).cuda()
            else:
                input_char = Variable(torch.stack(batch[0]))
                input_poke = Variable(torch.t(torch.stack(batch[1]).type(torch.FloatTensor)))

            output_char, output_poke = model(input_char, input_poke, 0) # turn off teacher forcing

            # trg = [trg len, batch size]
            # output = [trg len, batch size, output dim]

            output_char_dim = output_char.shape[-1]

            output_char = output_char[1:].view(-1, output_char_dim)
            input_char = input_char[1:].view(-1)

            # trg = [(trg len - 1) * batch size]
            # output = [(trg len - 1) * batch size, output dim]

            loss = criterion(output_char, input_char)
            loss_poke = emb_criterion(output_poke, input_poke)

            epoch_char_loss += loss.item()
            epoch_poke_loss += loss_poke.item()

    return epoch_char_loss / len(data_loader), epoch_poke_loss / len(data_loader)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs