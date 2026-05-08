import cv2
import numpy as np
import argparse
import os

from matplotlib import pyplot as plt
from pollenesia.utils import get_logger, get_file_list

logger = get_logger(__name__)


def contours_intersect(cnt1, cnt2):
    """Check if two contours intersect by comparing all edge points"""
    x1, y1, w1, h1 = cv2.boundingRect(cnt1)
    x2, y2, w2, h2 = cv2.boundingRect(cnt2)

    if x1 + w1 < x2 or x2 + w2 < x1 or y1 + h1 < y2 or y2 + h2 < y1:
        return False

    return True


def load_image(fname: str):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.bitwise_not(gray)

    window = gray.shape[0] // 11 - 1
    window += (window - 1) % 2
    threshold = 12
    threshold_brightness = threshold * 0.8

    background = cv2.GaussianBlur(gray, (window, window), 0)
    gray = cv2.subtract(gray, background)

    idx = gray < 4
    gray[idx] = np.mean(gray)
    # cv2.imshow('Prepared', gray)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    filtered_contours = []

    for cnt in contours:
        overlaps = False
        for filtered_cnt in filtered_contours:
            if contours_intersect(cnt, filtered_cnt):
                overlaps = True
                break

        if not overlaps:
            filtered_contours.append(cnt)

    n_particles = 0
    n_red = 0
    n_yellow = 0
    b = np.ndarray((0, 2))

    for cnt in filtered_contours:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        ratio = max(w / h, h / w)
        roi = gray[y:y+h, x:x+w]
        brightness = np.mean(roi)
        croi = img[y:y+h, x:x+w]
        color = np.mean(croi, axis=(0, 1))
        color /= np.max(color)
        b = np.vstack((b, [area, brightness]))
        if area >= 4 and area < 10000 and brightness > threshold_brightness and ratio < 4:
            n_particles += 1
            clr = np.clip(color * 255, 0, 255).astype(np.uint8)
            clr = tuple(int(x) for x in clr)
            if clr[2] - clr[1] > 64 and clr[2] - clr[0] > 64:
                clr = (0, 0, 255)
                n_red += 1
            elif (clr[1] + clr[2])/2 - clr[0] > 64 and abs(clr[1] - clr[2]) < 32:
                clr = (0, 255, 255)
                n_yellow += 1
            cv2.rectangle(img, (x, y), (x+w, y+h), clr, 1)

    logger.info(f'Particle Number: {n_particles}')
    logger.info(f'Red Particles: {n_red}')
    logger.info(f'Yellow Particles: {n_yellow}')

    # fig, ax = plt.subplots(1, 1)
    # ax.scatter(b[:, 0], b[:, 1])
    # ax.axhline(threshold_brightness, color='r', alpha=0.3)
    # ax.grid()
    # ax.set_xlabel('Area')
    # ax.set_ylabel('Brightness')
    # fig.set_size_inches(16, 9)
    # fig.set_tight_layout(True)
    # plt.show()

    # font = cv2.FONT_HERSHEY_DUPLEX
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    font_weight = 1
    text_color = (0, 255, 255)

    label = f'Particles: {n_particles:3d}'
    cv2.putText(img, label, (32, 32), font,
                font_scale, text_color, font_weight)
    label = f'Red: {n_red:3d}'
    cv2.putText(img, label, (32, 64), font,
                font_scale, text_color, font_weight)
    label = f'Yellow: {n_yellow:3d}'
    cv2.putText(img, label, (32, 96), font,
                font_scale, text_color, font_weight)

    ofname = os.path.basename(fname)
    dirname = os.path.dirname(fname)
    dirname = os.path.join(dirname, os.pardir)
    ofname = os.path.join(dirname, 'processed', ofname)
    ofname = os.path.relpath(ofname)
    cv2.imwrite(ofname, img)

    if False:
        gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        images = np.hstack((gray_color, img))
        cv2.imshow('Prepared', images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return n_particles, n_red, n_yellow


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
    n_red = []
    n_yellow = []
    for fname in flist:
        logger.info(fname)
        n, r, y = load_image(fname)
        n_particles.append(n)
        n_red.append(r)
        n_yellow.append(y)

    n_particles = np.array(n_particles)
    n_red = np.array(n_red)
    n_yellow = np.array(n_yellow)
    t = np.arange(n_particles.shape[0]) * 600.0 / 3600.0
    fig, (ax, ax2, ax3) = plt.subplots(3, 1)
    ax.plot(t, n_particles, '.-', color='tab:blue')
    ax2.plot(t, n_red, '.-', color='tab:red')
    ax3.plot(t, n_yellow, '.-', color='tab:orange')
    ax.grid()
    ax2.grid()
    ax3.grid()
    ax.set_xlabel('Hours')
    ax2.set_xlabel('Hours')
    ax3.set_xlabel('Hours')
    ax.set_ylabel('Particles')
    ax2.set_ylabel('Red Particles')
    ax3.set_ylabel('Yellow Particles')
    # ax.set_xlim(0, t[-1])
    ax.set_xlim(0, t[-1])
    ax2.set_xlim(0, t[-1])
    ax3.set_xlim(0, t[-1])
    fig.set_size_inches(16, 4.5)
    fig.set_tight_layout(True)
    plt.show()


if __name__ == '__main__':
    main()
