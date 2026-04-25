
from h5py import File, Group, Dataset
from matplotlib import pyplot
from scipy.constants import speed_of_light
from pollenesia.utils import get_logger, get_file_list
from scipy.signal import find_peaks

import argparse
import numpy as np
import os
import shutil

logger = get_logger(__name__)


def get_image_fname(fname: str):
    path = os.path.dirname(fname)
    f = File(fname, 'r')
    d = f['data']
    header = list(f['data'].attrs['header'])
    header: list
    i_sharpness = header.index('Sharpness')
    sharpness = d[:, i_sharpness]

    mean = np.mean(sharpness)
    mad = np.median(np.abs(sharpness - mean))
    threshold = mean + mad * 6
    i_peaks, _ = find_peaks(sharpness, height=threshold)

    if len(i_peaks) > 0:
        i_max_sharpness = np.min(i_peaks)
    else:
        i_max_sharpness = np.argmax(sharpness)
    info = f'mean={mean:.2f}, mad={mad:.2f}, threshold={threshold:.2f}, i_max_sharpness={i_max_sharpness}'
    logger.info(info)

    image_fname = f'image{i_max_sharpness:03d}.webp'
    image_path = os.path.join(path, image_fname)
    logger.info(image_path)
    f.close()
    return image_path


def copy_images(fnames: list[str], odir: str):
    for fname in fnames:
        dirname = os.path.basename(os.path.dirname(fname))
        dst = os.path.join(odir, f'{dirname}.webp')
        logger.info(dst)
        shutil.copyfile(fname, dst)


def remove_images(dirname: str):
    for fname in os.listdir(dirname):
        if fname.endswith('.webp'):
            os.remove(os.path.join(dirname, fname))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input', help='path', type=str)
    parser.add_argument('-o', '--output', help='output directory',
                        default='/var/log/pollenesia/filtered', type=str)

    try:
        options, _ = parser.parse_known_args()
        logger.info(vars(options))
    except Exception as e:
        logger.error(e)
        exit(0)

    pattern = options.input
    options.output = os.path.expanduser(options.output)
    if not os.path.exists(options.output):
        os.makedirs(options.output)
    odir = options.output

    flist = get_file_list(pattern, recursive=True)
    image_list = []

    for fname in flist:
        logger.info(fname)
        image = get_image_fname(fname)
        image_list.append(image)

    copy_images(image_list, odir)
    if False:
        dirname = os.path.dirname(pattern)
        flist = get_file_list(dirname, recursive=True)
        for d in flist:
            if os.path.isdir(d):
                remove_images(d)


if __name__ == '__main__':
    main()
