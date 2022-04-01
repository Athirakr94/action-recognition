import pandas as pd
import joblib
import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelBinarizer
import cv2
import time
# get all the image folder paths
all_paths = os.listdir('../input/data')
folder_paths = [x for x in all_paths if os.path.isdir('../input/data/' + x)]
print(f"Folder paths: {folder_paths}")
print(f"Number of folders: {len(folder_paths)}")
# we will create the data for the following labels, 
# add more to list to use those for creating the data as well
create_labels = ['springboard diving', 'surfing water', 'swimming backstroke','swimming breast stroke','swimming butterfly stroke']
# create a DataFrame
data = pd.DataFrame()
image_formats = ['jpg', 'png'] # we only want images that are in this format
labels = []
image_paths=[]
counter = 0
for i, folder_path in tqdm(enumerate(folder_paths), total=len(folder_paths)):
    print(folder_paths)
    if folder_path not in create_labels:
        print("Not found")
        continue
    image_paths = os.listdir('../input/data/'+folder_path)
    
#     # save image paths in the DataFrame
#     for videos in image_list:
#         cap= cv2.VideoCapture('../input/data/'+folder_path+"/"+videos)
# #         cap=cv2.VideoCapture(path)

#         # print(path,cap.isOpened())
#         count=1
#         time.sleep(0.1)
#         while(cap.isOpened()):            
#             ret, image = cap.read()
#             print(ret)
#             if ret:

# #                 # Saves the frames with frame-count
#                 res_name='../input/data/'+folder_path+"/"+"/"+videos.split(".")[0]+"_"+str(count)+".jpg"
#                 image_paths.append(res_name)
# #                 print("res name",res_name)
#                 cv2.imwrite(res_name, image)
#                 count+=1
#             else:
#                 break
#         os.remove('../input/data/'+folder_path+"/"+videos)
    for image_path in image_paths:
        label = folder_path
        print("Image",image_path)
        # if ".png" in image_path or ".jpg" in image_path:
        #     print("inside")
        data.loc[counter, 'image_path'] = '../input/data/'+folder_path+"/"+image_path
        labels.append(label)
        # print("Adding",label)
        counter += 1
print(labels)
labels = np.array(labels)
# one-hot encode the labels
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
if len(labels[0]) == 1:
    for i in range(len(labels)):
        index = labels[i]
        data.loc[i, 'target'] = int(index)
elif len(labels[0]) > 1:
    for i in range(len(labels)):
        index = np.argmax(labels[i])
        data.loc[i, 'target'] = int(index)
# shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)
print(f"Number of labels or classes: {len(lb.classes_)}")
print(f"The first one hot encoded labels: {labels[0]}")
print(f"Mapping the first one hot encoded label to its category: {lb.classes_[0]}")
print(f"Total instances: {len(data)}")
 
# save as CSV file
data.to_csv('../input/data_kinetics.csv', index=False)
 
# pickle the binarized labels
print('Saving the binarized labels as pickled file')
joblib.dump(lb, '../output/lb_kinetics.pkl')
 
print(data.head(5))