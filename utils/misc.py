import json
from os.path import join
import numpy as np
from PIL import Image
from utils.isoutil import *
import torch
import torchvision
import sys


class MeanTracker(object):
    def __init__(self, name):
        self.values = []
        self.name = name

    def add(self, val):
        self.values.append(float(val))

    def mean(self):
        return np.mean(self.values)

    def flush(self):
        mean = self.mean()
        self.values = []
        return self.name, mean

def save_params(config, training_path):
    save_dict_path = join(training_path, 'param.json')
    with open(save_dict_path, 'w') as outfile:
        json.dump(config,
                   outfile,
                   sort_keys=False,
                   indent=4,
                   separators=(',', ': '))

def load_params(config_file):
    with open(config_file, 'r') as f:
        data = json.load(f)
    return data


def save_images(isovists, iter_num, title, sample_folder):
    figs=[]
    for i, x_ in enumerate(isovists):
        x_ = np.squeeze(x_)
        figs.append(plot_isovist_numpy(x_, figsize=(1,1)))
    figs = torch.tensor(figs, dtype=torch.float)
    nrow = int(np.sqrt(isovists.shape[0]))
    im = torchvision.utils.make_grid(figs, normalize=True, range=(0, 255), nrow=nrow)
    im = Image.fromarray(np.uint8(np.transpose(im.numpy(), (1, 2, 0))*255))
    im.save(join(sample_folder, f'{title}_{iter_num:06}.jpg'))


def imshow(img):
    npimg = img.numpy()
    plt.figure(figsize = (30,30))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


def write(text):
    sys.stdout.write('\n' + text)
    if hasattr(sys.stdout, 'flush'):
        sys.stdout.flush()


def get_batches(isovist, batch_num):
    batches = []
    for i in range(0, len(isovist), batch_num):
        batches.append(isovist[i:i+batch_num])
    return batches


def get_data(df):
    scale = df.at[0, 'scale']
    isovist_rad = df.at[0, 'isovist_rad']
    data = (df.to_numpy()).astype(np.float32)
    loc = data[:, 2:4]*scale
    isovist = data[:,4:]
    return scale, isovist_rad, loc, isovist


def get_polylines(file_path):
    with open(file_path) as f:
        lines = f.readlines()
    
    polylines = []
    r = 1000
    for pointlist in lines:
        points = pointlist.split('|')
        polyline = [(float(k.split(',')[0])*r, float(k.split(',')[1])*r) for k in points]
        polylines.append(polyline)
    return polylines



