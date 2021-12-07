# importing two required module
import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('_mpl-gallery')


model=["MobileNetv3","MobileNetv3_320","ResNet50","YOLOv5s"]
# Creating a numpy array
inferencetime = np.array([0.0874, 0.0495, 0.3238, 0.033])
accuracy = np.array([87.4, 27.3, 86.1, 85.1])
# Plotting point using sactter method
plt.figure(figsize=(8,6))
plt.scatter(inferencetime,accuracy,s=100,color="maroon")
plt.grid()
plt.xlim([0, 0.4])
plt.ylim([25, 95])
plt.xlabel("Inference time [s]", fontsize = 14)
plt.ylabel("mean Averaged Precision (mAP) [%]", fontsize = 14)
plt.title("Accuracy vs inference time",fontsize=15)
plt.annotate(model[0], (inferencetime[0] - 0.01, accuracy[0] + 1.5), fontsize = 14)
plt.annotate(model[1], (inferencetime[1] + 0.01, accuracy[1] - 0.01), fontsize = 14)
plt.annotate(model[2], (inferencetime[2] - 0.05, accuracy[2] - 3.5), fontsize = 14)
plt.annotate(model[3], (inferencetime[3], accuracy[3] - 3.5), fontsize = 14)
plt.savefig('plots/accvsinf.png', pad_inches = 0.1)
plt.show()
