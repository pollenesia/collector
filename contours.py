import cv2
import numpy as np
import argparse
import os

from matplotlib import pyplot as plt
from utils import get_logger, get_file_list

logger = get_logger(__name__)


def load_image(fname: str):
    # 'data/filtered/250809142446.jpg'
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)

    window = gray.shape[0] // 11 - 1
    window += (window - 1) % 2
    threshold = 60
    threshold_brightness = threshold * 0.8

    background = cv2.GaussianBlur(gray, (window, window), 0)
    gray = cv2.subtract(gray, background)

    idx = gray < 24
    gray[idx] = np.mean(gray)
    # cv2.imshow('Prepared', gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    n_particles = 0
    b = np.ndarray((0, 2))
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = max(w / h, h / w)
        roi = gray[y:y+h, x:x+w]
        brightness = np.mean(roi)
        b = np.vstack((b, [area, brightness]))
        if area >= 4 and area < 1000 and brightness > threshold_brightness and ratio < 3:
            n_particles += 1
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 1)

    logger.info(f'Particle Number: {n_particles}')
    logger.info(f'Particle Shape: {b.shape}')

    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(b[:, 0], b[:, 1])
    # ax.axhline(threshold_brightness, color='r', alpha=0.3)
    # ax.grid()
    # ax.set_xlabel('Area')
    # ax.set_ylabel('Brightness')
    # fig.set_size_inches(16, 9)
    # fig.set_tight_layout(True)
    # plt.show()

    font = cv2.FONT_HERSHEY_DUPLEX
    # text,coordinate,font,size of text,color,thickness of font
    label = f'Particle Number: {n_particles:2d}'
    cv2.putText(img, label, (32, 32), font, 1, (0, 255, 255), 1)

    ofname = os.path.basename(fname)
    dirname = os.path.dirname(fname)
    dirname = os.path.join(dirname, os.pardir)
    ofname = os.path.join(dirname, 'processed', ofname)
    ofname = os.path.relpath(ofname)
    cv2.imwrite(ofname, img)

    # images = np.hstack((gray_color, img))
    # cv2.imshow('Prepared', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return n_particles


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
    flist = get_file_list(pattern, recursive=True)

    n_particles = []
    for fname in flist:
        logger.info(fname)
        n = load_image(fname)
        n_particles.append(n)

    n_particles = np.array(n_particles)
    t = np.arange(n_particles.shape[0]) * 600.0 / 3600.0
    fig, ax = plt.subplots(1, 1)
    ax.plot(t, n_particles, '.-')
    ax.grid()
    ax.set_xlabel('Hours')
    ax.set_ylabel('Particle Number')
    # ax.set_xlim(0, t[-1])
    ax.set_xlim(0)
    fig.set_size_inches(16, 4.5)
    fig.set_tight_layout(True)
    plt.show()


if __name__ == '__main__':
    main()
