from __future__ import print_function
import torch
import torchvision.transforms as transforms

import argparse
import os
import random
import sys
import pprint
import datetime
import dateutil.tz
import time


dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


from miscc.config import cfg, cfg_from_file


# 19 classes --> 7 valid classes with 8,555 images
DOG_LESS = ['n02084071', 'n01322604', 'n02112497', 'n02113335', 'n02111277',
            'n02084732', 'n02111129', 'n02103406', 'n02112826', 'n02111626',
            'n02110958', 'n02110806', 'n02085272', 'n02113978', 'n02087122',
            'n02111500', 'n02110341', 'n02085374', 'n02084861']
# 118 valid classes with 147,873 images
DOG = ['n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240',
       'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094',
       'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078',
       'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721',
       'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635',
       'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428',
       'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114',
       'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889',
       'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585',
       'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474',  # 10
       'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267',
       'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236',
       'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388',
       'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480',
       'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162',
       'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855',
       'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662',
       'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908',
       'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915',
       'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185',  # 20
       'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129',
       'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137',
       'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624',
       'n02113712', 'n02113799', 'n02113978']
# 17 classes --> 5 classes with 6500 images
CAT = ['n02121808', 'n02124075', 'n02123394', 'n02122298', 'n02123159',
       'n02123478', 'n02122725', 'n02123597', 'n02124484', 'n02124157',
       'n02122878', 'n02123917', 'n02122510', 'n02124313', 'n02123045',
       'n02123242', 'n02122430']
CLASS_DIC = {'dog': DOG, 'cat': CAT}


def parse_args():
    tp = lambda x: list(map(float, x.split(',')))
    itp = lambda x: list(map(int, x.split(',')))
    parser = argparse.ArgumentParser(description='Train a GAN network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/birds_proGAN.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=str, default='-1')
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--d_learning_rate', dest='d_learning_rate', type=tp, default='0.0002')
    parser.add_argument('--g_learning_rate', dest='g_learning_rate', type=tp, default='0.0002')
    parser.add_argument('--uncond_loss', dest='uncond_loss', type=tp, default='1.0')
    parser.add_argument('--df_dim', dest='df_dim', type=itp, default='64')
    parser.add_argument('--gf_dim', dest='gf_dim', type=itp, default='64')
    parser.add_argument('--z_dim', dest='z_dim', type=itp, default='100')
    parser.add_argument('--r_num', dest='r_num', type=itp, default='2')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != '-1':
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        if cfg.DATASET_NAME == 'birds':
            bshuffle = False
            split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM-1))
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.DATA_DIR.find('lsun') != -1:
        from datasets import LSUNClass
        dataset = LSUNClass('%s/%s_%s_lmdb' %
                            (cfg.DATA_DIR, cfg.DATASET_NAME, split_dir),
                            base_size=cfg.TREE.BASE_SIZE, transform=image_transform)
    elif cfg.DATA_DIR.find('imagenet') != -1:
        from datasets import ImageFolder
        dataset = ImageFolder(cfg.DATA_DIR, split_dir='train',
                              custom_classes=CLASS_DIC[cfg.DATASET_NAME],
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    elif cfg.GAN.B_CONDITION:  # text to image task
        from datasets import TextDataset
        dataset = TextDataset(cfg.DATA_DIR, split_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
    elif cfg.GAN.POKEMON:  # pokemon to image task
        from datasets import PokemonDataset
        dataset = PokemonDataset(cfg.DATA_DIR, split_dir,
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)

    assert dataset
    num_gpu = len(cfg.GPU_ID.split(','))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE * num_gpu,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    if not cfg.GAN.B_CONDITION and not cfg.GAN.POKEMON:
        from trainer import GANTrainer as trainer
    else:
        if not cfg.GAN.POKEMON:
            from trainer import condGANTrainer as trainer
        else:
            from trainer import pokemonTrainer as trainer

    cfg.TRAIN.MAX_EPOCH = 1
    grid_search_result='d_lr,g_lr,unc_l,df_dim,gf_dim,z_dim,r_num,errD,errG,KL_loss\n'
    for d_learning_rate in args.d_learning_rate:
        cfg.TRAIN.DISCRIMINATOR_LR = d_learning_rate
        for g_learning_rate in args.g_learning_rate:
            cfg.TRAIN.GENERATOR_LR = g_learning_rate
            for uncond_loss in args.uncond_loss:
                cfg.TRAIN.COEFF.UNCOND_LOSS = uncond_loss
                for df_dim in args.df_dim:
                    cfg.GAN.DF_DIM = df_dim
                    for gf_dim in args.gf_dim:
                        cfg.GAN.GF_DIM = gf_dim
                        for z_dim in args.z_dim:
                            cfg.GAN.Z_DIM = z_dim
                            for r_num in args.r_num:
                                cfg.GAN.R_NUM = r_num
                                algo = trainer(output_dir, dataloader, imsize)
                                errD, errG, kl_loss = algo.train_once()
                                grid_search_result += '%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' % (
                                d_learning_rate, g_learning_rate, uncond_loss, df_dim, gf_dim,
                                z_dim, r_num, errD, errG, kl_loss)

    with open("result.csv","w") as f:
        f.write(grid_search_result)
