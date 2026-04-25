from h5py import File
import os
import numpy as np

from pollenesia.utils import get_file_list
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

fname = '~/tmp/2/**/data.h5'
fname = os.path.expanduser(fname)
flist = get_file_list(fname)

for fname in flist:
    f = File(fname, 'r')
    d = f['data']
    sharpness = d[:, 2]
    mean = np.mean(sharpness)
    mad = np.median(np.abs(sharpness - mean))
    threshold = mean + mad * 6
    i_peaks, properties = find_peaks(sharpness, height=threshold)

    if len(i_peaks) > 0:
        i_max_sharpness = np.min(i_peaks)
    else:
        i_max_sharpness = np.argmax(sharpness)

    path = os.path.dirname(fname)
    image_fname = f'image{i_max_sharpness:03d}.webp'
    image_path = os.path.join(path, image_fname)
    print(image_path)

    if False:
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(16, 9)
        fig.set_tight_layout(True)
        ax1.plot(sharpness, '.-')
        ax1.plot(i_peaks, sharpness[i_peaks], 'x')
        ax1.plot(i_max_sharpness, sharpness[i_max_sharpness], '^')
        ax1.axhline(threshold, color='r', alpha=0.3)
        plt.show()
