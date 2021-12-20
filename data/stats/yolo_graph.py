import matplotlib.pyplot as plt
import pickle
import numpy as np


filename = 'yolo_map_loss.txt'
with open(filename, 'r') as f:
    data = f.readlines()


losses = []
acc = []
epoch = []
for i in range(0, len(data) - 2, 5):
    currLoss = data[i+1].split()[4]
    currEpoch = data[i+1].split()[0][:-3]
    currAcc = data[i+3].split()[5]
    losses.append(float(currLoss))
    epoch.append(int(currEpoch))
    acc.append(float(currAcc))

losses = np.array(losses)
acc = np.array(acc)*100
epoch = np.array(epoch)

fig, axs = plt.subplots(2,1)
axs[0].plot(epoch, acc, 'maroon')
axs[0].set_ylabel("mean Averaged Precision\n (mAP) [%]", fontsize = 10)
axs[0].grid(True)


axs[1].plot(epoch, losses, 'maroon')
axs[1].set_ylabel("Training loss", fontsize = 10)
axs[1].grid(True)

axs[1].set_xlabel("Epoch", fontsize = 14)
fig.suptitle("Training results for 100 epochs with YOLOv5s",fontsize=15)


plt.savefig('trainingacclossYOLO.png', pad_inches = 0.1)
plt.show()
