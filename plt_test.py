import matplotlib

import matplotlib.pyplot as plt

import numpy as np

# データの用意
x = np.arange(0, 10, 0.1)
y = np.sin(x)
# グラフの描画
plt.plot(x, y)
plt.savefig('figure01.jpg')