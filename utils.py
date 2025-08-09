import glob
import logging

LOG_FORMAT = f"%(asctime)s [%(levelname)s] %(filename)s %(funcName)s(%(lineno)d): %(message)s"


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(stream_handler)
    return logger


def get_file_list(pattern, recursive=False):
    flist = [fname for fname in glob.glob(pattern, recursive=recursive)]
    flist = sorted(flist)
    return flist
