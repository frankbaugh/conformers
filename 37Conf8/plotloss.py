import matplotlib.pyplot as plt
import numpy as np

train_loss = np.load('train_loss.npy')
test_loss = np.load('test_loss.npy')

plt.plot(test_loss)
plt.plot(train_loss)
plt.show()