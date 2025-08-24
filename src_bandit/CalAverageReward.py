import numpy as np
import matplotlib.pyplot as plt

from Bandit import Bandit

band = Bandit()
print(band.rates[0])

Q = 0
times = 100000
Q_list = []
for n in range(1, times):
    reward = band.play(0)
    Q += (reward - Q) / n
    Q_list.append(Q)

plt.plot([i for i in range(1, times)],
         Q_list,
         color='red',
         alpha=0.6,  # 透明度
         label='散点')
plt.show()
