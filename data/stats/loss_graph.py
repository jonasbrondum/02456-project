import pickle
import numpy as np
import matplotlib.pyplot as plt


def stats_loader(model):
    pickle_off = open (model, "rb")
    stats = pickle.load(pickle_off)[2:-2].split(',')
    loss = stats[0:10]
    acc = stats[10:]
    loss[9] = loss[9][:-1]
    acc[0] = acc[0][2:]
    loss = np.array([float(x) for x in loss])
    acc = np.array([float(x) for x in acc])
    return loss, acc

# mobilenet large 'model_stat_mobilnetv3large.txt'
# mobilenet large 320 'model_stat_mobilenetv3large_320.txt'
# resnet "resnet50_10_epoch_model_stat.txt"
model_stats = ["resnet50_10_epoch_model_stat.txt", 'model_stat_mobilnetv3large.txt', 'model_stat_mobilenetv3large_320.txt']
artists = ["ResNet50","MobileNetv3","MobileNetv3_320"] # ,"YOLOv5s"

losses = []
acc = []
for i, model in enumerate(model_stats):
    temploss, tempacc = stats_loader(model)
    losses.append(temploss)
    acc.append(tempacc)

losses = np.array(losses)
# Accuracy converted to "percent"
acc = np.array(acc)*100

colorscheme = ['darkblue','maroon','darkgreen','lightblue']

# plt.figure(figsize=(8,6))
fig, axs = plt.subplots(2,1)
for i, (modelacc, modelloss) in enumerate(zip(acc, losses)):
    x = np.arange(1,len(modelacc)+1)
    axs[0].plot(x, modelacc, colorscheme[i])
    axs[0].set_ylabel("mean Averaged Precision\n (mAP) [%]", fontsize = 10)
    axs[0].grid(True)
    axs[0].legend(artists)

    axs[1].plot(x, modelloss, colorscheme[i])
    axs[1].set_ylabel("Training loss", fontsize = 10)
    axs[1].grid(True)

axs[1].set_xlabel("Epoch", fontsize = 14)
fig.suptitle("Training results for 10 epochs",fontsize=15)
# plt.legend(artists)
plt.savefig('trainingaccloss.png', pad_inches = 0.1)

plt.show()
