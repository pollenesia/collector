
from email.mime import image

import psycopg2
import base64
import time
import os
import argparse
import cv2
import rawpy
import numpy as np

from datetime import datetime, timedelta
from PIL import Image
from pollenesia.utils import get_file_list, get_logger
from pollenesia.contours import load_image
from pollenesia.utils import get_logger, get_file_list

logger = get_logger(__name__)

DB_NAME = 'pollenesia'
DB_USER = 'pollenesia'
DB_PASS = 'pollenesia'
DB_HOST = 'localhost'
DB_PORT = '5433'

conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER,
                        password=DB_PASS, host=DB_HOST, port=DB_PORT)
cursor = conn.cursor()

create_table_query = '''
CREATE TABLE IF NOT EXISTS images (
    timestamp BIGINT PRIMARY KEY,
    image_data TEXT NOT NULL
);
'''
cursor.execute(create_table_query)


create_table_metrics = '''
CREATE TABLE IF NOT EXISTS metrics (
    timestamp BIGINT PRIMARY KEY,
    n_particles INTEGER NOT NULL,
    n_red INTEGER NOT NULL,
    n_yellow INTEGER NOT NULL
);
'''
cursor.execute(create_table_metrics)


def insert_data(timestamp, image_data):
    insert_query = '''
        INSERT INTO images (timestamp, image_data) VALUES (%s, %s) 
        ON CONFLICT (timestamp) DO UPDATE SET image_data = EXCLUDED.image_data;
    '''
    cursor.execute(insert_query, (timestamp, image_data))


def insert_metrics(timestamp, n_particles, n_red, n_yellow):
    insert_query = '''
        INSERT INTO metrics (timestamp, n_particles, n_red, n_yellow) VALUES (%s, %s, %s, %s)
        ON CONFLICT (timestamp) DO UPDATE SET n_particles = EXCLUDED.n_particles, n_red = EXCLUDED.n_red, n_yellow = EXCLUDED.n_yellow;
    '''
    cursor.execute(insert_query, (timestamp, n_particles, n_red, n_yellow))


def basename2unixtime(basename: str) -> int:
    ts = datetime.strptime(basename, '%y%m%d%H%M%S')
    return int(ts.timestamp())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path', type=str)

    try:
        options, _ = parser.parse_known_args()
        logger.info(vars(options))
    except Exception as e:
        logger.error(e)
        exit(0)

    pattern = options.path
    pattern = os.path.expanduser(pattern)
    flist = get_file_list(pattern, recursive=True)

    for fname in flist:
        logger.info(fname)
        basename = os.path.basename(fname)
        basename = os.path.splitext(basename)[0]
        timestamp = basename2unixtime(basename)
        raw = rawpy.imread(fname)

        rgb = raw.postprocess(
            # use_camera_wb=True,
            output_bps=16,
            no_auto_bright=True,
            use_auto_wb=True,
        )

        rgb = rgb.astype(np.float32) / 65535.0
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v = hsv[:, :, 2]
        threshold = np.percentile(v, 50)
        intercept = -threshold / (1.0 - threshold)
        slope = 1.0 / (1.0 - threshold)
        v = slope * v + intercept
        v = np.clip(v, 0.0, 1.0)
        hsv[:, :, 2] = v

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr = np.clip(bgr * 255.0, 0, 255).astype(np.uint8)

        cv2.imshow('DNG Image', bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        continue

        n_particles, n_red, n_yellow = load_image(fname)
        insert_metrics(timestamp, n_particles, n_red, n_yellow)

        with open(fname, 'rb') as f:
            image_data = f.read()
            base64_image_data = 'data:image/webp;base64,'
            base64_image_data += base64.b64encode(image_data).decode('utf-8')
            insert_data(timestamp, base64_image_data)

    conn.commit()
    cursor.close()
    conn.close()


if __name__ == '__main__':
    main()
