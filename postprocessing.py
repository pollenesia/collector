
from h5py import File, Group, Dataset
from matplotlib import pyplot
from scipy.constants import speed_of_light
from utils import get_logger, get_file_list

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
    i_max_sharpness = np.argmax(d[:, i_sharpness])
    image_fname = f'image{i_max_sharpness:03d}.jpg'
    image_path = os.path.join(path, image_fname)
    logger.info(image_path)
    f.close()
    return image_path


def copy_images(fnames: list[str]):
    for fname in fnames:
        dirname = os.path.basename(os.path.dirname(fname))
        dst = os.path.join('data', 'filtered', f'{dirname}.jpg')
        logger.info(dst)
        shutil.copyfile(fname, dst)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path', type=str)

    try:
        options, _ = parser.parse_known_args()
        logger.info(vars(options))
    except Exception as e:
        logger.error(e)
        exit(0)

    # os.path.join(options.path, '**', 'data.h5')
    pattern = options.path
    flist = get_file_list(pattern, recursive=True)
    image_list = []

    for fname in flist:
        logger.info(fname)
        image = get_image_fname(fname)
        image_list.append(image)

    copy_images(image_list)

    # f = open('list.txt', 'wt')
    # for line in image_list:
    #     f.write(f'{line}\n')
    # f.close()


if __name__ == '__main__':
    main()
