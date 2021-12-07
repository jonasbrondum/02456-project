# importing two required module
import numpy as np
import matplotlib.pyplot as plt

# plt.style.use('_mpl-gallery')


annotations=["MobileNetv3","MobileNetv3_320","ResNet50","YOLOv5"]
# Creating a numpy array
inferencetime = np.array([0.052, 0.05, 0.3238, 0.03])
accuracy = np.array([65.3, 61.2, 86.1, 85.1])
# Plotting point using sactter method
plt.figure(figsize=(8,6))
plt.scatter(inferencetime,accuracy,s=100,color="maroon")
plt.grid()
plt.xlabel("Inference time [s]")
plt.ylabel("mean Averaged Precision [%]")
plt.title("Accuracy vs inference time",fontsize=15)
for i, label in enumerate(annotations):
    plt.annotate(label, (inferencetime[i], accuracy[i]))
plt.savefig('plots/accvsinf.png', pad_inches = 0.1)
plt.show()
