#!-*- coding:utf-8 -*-
"""image processing utils"""
from imageio import imread
from imageio import imsave
import numpy as np

def get_image_from_path(path):
    img = imread(path).astype(np.float32)/255.0
    return img

def save_image_to_path(path,img):
    imsave(path,(255.0*img).astype(np.uint8))