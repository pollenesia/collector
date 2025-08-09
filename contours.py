import cv2
import numpy as np
from matplotlib import pyplot as plt


img = cv2.imread('data/filtered/250809142446.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)

window = gray.shape[0] // 8 - 1
threshold = 48

background = cv2.GaussianBlur(gray, (window, window), 0)
gray = cv2.subtract(gray, background)
gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

_, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)

contours, _ = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

n_particles = 0
b = np.ndarray((0, 2))
for cnt in contours:
    area = cv2.contourArea(cnt)
    x, y, w, h = cv2.boundingRect(cnt)
    roi = gray[y:y+h, x:x+w]
    brightness = np.mean(roi)
    b = np.vstack((b, [area, brightness]))
    print(brightness)
    if area >= 4 and area < 500 and brightness > threshold:
        n_particles += 1
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 1)

gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
images = np.hstack((gray_color, img))
print(f'Particle Number: {n_particles}')
print(f'Particle Shape: {b.shape}')

fig, ax = plt.subplots(1, 1)
ax.scatter(b[:, 0], b[:, 1])
# ax.axvline(threshold, np.min(b[:, 0]), np.max(b[:, 0]), colors='r')
ax.axhline(threshold, color='r', alpha=0.3)
ax.grid()
ax.set_xlabel('Area')
ax.set_ylabel('Brightness')
fig.set_size_inches(16, 9)
fig.set_tight_layout(True)
plt.show()

cv2.imshow('Prepared', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()
