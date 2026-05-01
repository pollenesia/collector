from h5py import File
import os
import numpy as np

from pollenesia.utils import get_file_list, get_logger
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

logger = get_logger(__name__)

fname = '~/tmp/2/data.h5'
fname = os.path.expanduser(fname)
flist = get_file_list(fname)

for fname in flist:
    f = File(fname, 'r')
    d = f['data']
    header = list(f['data'].attrs['header'])
    header: list
    i_sharpness = header.index('FocusFoM')
    sharpness = d[:, i_sharpness]

    # sharpness = d[:, 2]
    mean = np.mean(sharpness)
    mad = np.median(np.abs(sharpness - mean))
    threshold = mean + mad * 6
    i_peaks, properties = find_peaks(sharpness, height=threshold)

    if len(i_peaks) > 0:
        i_max_sharpness = np.min(i_peaks)
    else:
        i_max_sharpness = np.argmax(sharpness)

    i_range_mm = header.index('Position_mm')
    range_mm = d[i_max_sharpness, i_range_mm]
    info = f'mean={mean:.2f}, mad={mad:.2f}, threshold={threshold:.2f}'
    logger.info(info)
    info = f'i_max_sharpness={i_max_sharpness}, range_mm={range_mm:.2f}'
    logger.info(info)

    path = os.path.dirname(fname)
    image_fname = f'image{i_max_sharpness:03d}.webp'
    image_path = os.path.join(path, image_fname)
    print(image_path)

    if True:
        fig, ax1 = plt.subplots(1, 1)
        fig.set_size_inches(16, 9)
        fig.set_tight_layout(True)
        ax1.plot(sharpness, '.-')
        ax1.plot(i_peaks, sharpness[i_peaks], 'x')
        ax1.plot(i_max_sharpness, sharpness[i_max_sharpness], '^')
        ax1.axhline(threshold, color='r', alpha=0.3)
        plt.show()
