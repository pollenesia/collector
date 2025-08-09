
from h5py import File, Group, Dataset
from matplotlib import pyplot
from scipy.constants import speed_of_light
from utils import get_logger, get_file_list

import argparse
import numpy as np
import os


logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path list', type=str, nargs='*')

    try:
        options, _ = parser.parse_known_args()
        logger.info(vars(options))
    except Exception as e:
        logger.error(e)
        exit(0)

    for i, path in enumerate(reversed(options.path)):
        pattern = os.path.join(path, '**', 'stat.txt')
        flist = get_file_list(pattern, recursive=True)

        for fname in flist:
            logger.info(fname)
            path0 = os.path.dirname(fname)
            rcv_name = load_replies(path0).lower()
            label = f'{i + 1}_{rcv_name}'
            d = read_stat_result(fname)
            write_result_json(path0, d)


if __name__ == '__main__':
    main()
