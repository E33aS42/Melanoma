import fastai
from fastai import *
from fastai.vision import *
import matplotlib.pyplot as plt
from pathlib import PosixPath
from PIL import Image
import os

path = PosixPath('images/train')

tfms = get_transforms(do_flip=False, flip_vert=True, max_rotate=90, max_zoom=0.5)
data = ImageDataBunch.from_folder(path, ds_tfms=tfms, size=224)

learn = cnn_learner(data, models.resnet34, metrics=accuracy)
learn.fit(5)

# save model
learn.export('mel_res34_size224.pkl')