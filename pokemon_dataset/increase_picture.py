#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# usage: ./increase_picture.py hogehoge.jpg
#

import cv2
import numpy as np
import math
import sys
import os


# ヒストグラム均一化
def equalizeHistRGB(src):
    RGB = cv2.split(src)
    Blue = RGB[0]
    Green = RGB[1]
    Red = RGB[2]
    for i in range(3):
        cv2.equalizeHist(RGB[i])

    img_hist = cv2.merge([RGB[0], RGB[1], RGB[2]])
    return img_hist


# ガウシアンノイズ
def addGaussianNoise(src):
    row, col, ch = src.shape
    mean = 0
    var = 0.1
    sigma = 5
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = src + gauss

    return noisy


# salt&pepperノイズ
def addSaltPepperNoise(src):
    row, col, ch = src.shape
    s_vs_p = 0.1
    amount = 0.008
    out = src.copy()
    # Salt mode
    num_salt = np.ceil(amount * src.size * s_vs_p)
    coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in src.shape]
    out[coords[:-1]] = (255, 255, 255)

    # Pepper mode
    # num_pepper = np.ceil(amount * src.size * (1. - s_vs_p))
    # coords = [np.random.randint(0, i - 1, int(num_pepper))
    #           for i in src.shape]
    # out[coords[:-1]] = (0, 0, 0)
    return out


# 画像をずらす
def moveImage(src):
    r = math.pi
    move_ms = [
                np.float32([[1, 0, 5], [0, 1, 0]]) ,
                np.float32([[1, 0, -5], [0, 1, 0]]) ,
                np.float32([[1, 0, 0], [0, 1, 5]]) ,
                np.float32([[1, 0, 0], [0, 1, -5]]) ,
                np.float32([[math.cos(r/40), math.sin(r/40), -5], [-math.sin(r/40), math.cos(r/40), 5]]) ,
                np.float32([[math.cos(-r/40), math.sin(-r/40), 5], [-math.sin(-r/40), math.cos(-r/40), -5]]) ,
    ]
    row, col, ch = src.shape
    out_images = []
    for m in move_ms:
        out_images.append(cv2.warpAffine(src, m, (row, col),borderValue=(255, 255, 255)))
    return out_images


if __name__ == '__main__':
    img_paths = os.listdir(sys.argv[1])
    for img_name in img_paths:
        imp_path = os.path.join(sys.argv[1], img_name)
        # ルックアップテーブルの生成
        min_table = 25
        max_table = 235
        diff_table = max_table - min_table
        gamma1 = 0.85
        gamma2 = 1.5

        LUT_HC = np.arange(256, dtype='uint8')
        LUT_LC = np.arange(256, dtype='uint8')
        LUT_G1 = np.arange(256, dtype='uint8')
        LUT_G2 = np.arange(256, dtype='uint8')

        LUTs = []

        # 平滑化用
        average_square = (2, 2)

        # ハイコントラストLUT作成
        for i in range(0, min_table):
            LUT_HC[i] = 0

        for i in range(min_table, max_table):
            LUT_HC[i] = 255 * (i - min_table) / diff_table

        for i in range(max_table, 255):
            LUT_HC[i] = 255

        # その他LUT作成
        for i in range(256):
            LUT_LC[i] = min_table + i * (diff_table) / 255
            LUT_G1[i] = 255 * pow(float(i) / 255, 1.0 / gamma1)
            LUT_G2[i] = 255 * pow(float(i) / 255, 1.0 / gamma2)

        LUT_LC[0] = 0
        LUTs.append(LUT_HC)
        # LUTs.append(LUT_LC)
        LUTs.append(LUT_G1)
        LUTs.append(LUT_G2)

        # 画像の読み込み
        img_src = cv2.imread(imp_path, 1)
        trans_img = []
        trans_img.append(img_src)

        # LUT変換
        for i, LUT in enumerate(LUTs):
            trans_img.append(cv2.LUT(img_src, LUT))

        # 平滑化
        trans_img.append(cv2.blur(img_src, average_square))

        # ヒストグラム均一化
        trans_img.append(equalizeHistRGB(img_src))

        # ノイズ付加
        trans_img.append(addGaussianNoise(img_src))
        trans_img.append(addSaltPepperNoise(img_src))

        # 画像を移動
        trans_img.extend(moveImage(img_src))

        # 反転
        flip_img = []
        for img in trans_img:
            flip_img.append(cv2.flip(img, 1))
        trans_img.extend(flip_img)

        # 保存
        if not os.path.exists("trans_images"):
            os.mkdir("trans_images")

        base = os.path.splitext(os.path.basename(imp_path))[0] + "_"
        img_src.astype(np.float64)
        for i, img in enumerate(trans_img):
            # 比較用
            # cv2.imwrite("trans_images/" + base + str(i) + ".jpg" ,cv2.hconcat([img_src.astype(np.float64), img.astype(np.float64)]))
            cv2.imwrite("trans_images/" + base + str(i) + ".jpg", img)
