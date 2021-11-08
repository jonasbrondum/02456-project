########## split data into train and test folder ##################
import os
import shutil 

#setwd()
cwd = os.getcwd()
data_path = cwd
data = os.listdir(data_path)


# Split files into train and test
from sklearn.model_selection import train_test_split

train, test = train_test_split(data, test_size=0.2, random_state=1)

# Create test and train directory
train_path = cwd + "/data/video1/train"
test_path = cwd + "/data/video1/test"

try:
   os.mkdir(train_path)
   os.mkdir(test_path)
except FileExistsError:
   pass

# Move train files to train folder
for file_name in train:
   shutil.copy(os.path.join(data_path, file_name), train_path)

# Move test files to test folder
for file_name in test:
   shutil.copy(os.path.join(data_path, file_name), test_path)

print("Size train data: ", len(train), "\n","Size test data: ", len(test))
print(len(data))

