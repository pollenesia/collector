from h5py import File
import os
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks

fname = '~/tmp/2/260422072438/data.h5'
fname = os.path.expanduser(fname)

f = File(fname, 'r')
d = f['data']
print(d.shape)
sharpness = d[:, 2]
# sharpness = d[:, -3]
threshold = 600
mean = np.mean(sharpness)
mad = np.median(np.abs(sharpness - mean))
threshold = mean + mad * 6
peaks_idx, properties = find_peaks(sharpness, height=threshold)
max_idx = np.min(peaks_idx) if len(peaks_idx) > 0 else np.argmax(sharpness)
fig, ax1 = plt.subplots(1, 1)
fig.set_size_inches(16, 9)
fig.set_tight_layout(True)
ax1.plot(sharpness, '.-')
ax1.plot(peaks_idx, sharpness[peaks_idx], 'x')
ax1.plot(max_idx, sharpness[max_idx], '^')

ax1.axhline(threshold, color='r', alpha=0.3)
plt.show()
