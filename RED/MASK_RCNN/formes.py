from mrcnn.model import log
from mrcnn import visualize
from mrcnn import utils
from mrcnn.config import Config
import os
import sys
import random
import math
import re
import time
from matplotlib import axes
import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt


ROOT_DIR = os.path.abspath("mrcnn")

sys.path.append(ROOT_DIR)


MODEL_DIR = os.path.join(ROOT_DIR, "logs")
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class ShapesConfig(Config):
    NAME = "shapes"

    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    NUM_CLASSES = 1 + 3  # background + 3 shapes

    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    TRAIN_ROIS_PER_IMAGE = 32

    STEPS_PER_EPOCH = 100

    VALIDATION_STEPS = 5


config = ShapesConfig()
config.display()


def get_ax(rows=1, cols=1, size=8):
    _, ax = plt.subplots(rows, cols, figzie=(size*cols, size*rows))
    return ax


class ShapesDataset(utils.Dataset):

    def load_shapes(self, count, heigth, width):
        self.add_class("shpaes", 1, "square")
        self.add_class("shpaes", 2, "circle")
        self.add_class("shpaes", 3, "triangle")
