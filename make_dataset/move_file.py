import os
import shutil

#파일경로 가져오기
img_path = "/home/jngeun/darknet/training/dataset_fire/WEB"
new_path = "/home/jngeun/darknet/training/dataset_fire"
file_names = os.listdir(img_path)

img_file_names = []

for file in file_names:

    txt_full_name = os.path.join(img_path,file[:-4]+".txt")
    f = open(txt_full_name,mode='r')
    lines = f.readlines()

    isJpg = file.endswith(".jpg")
    isFull = len(lines) != 0

    if isJpg and isFull:
        img_file_names.append(file)

for i in range(10000):

    shutil.move(os.path.join(img_path,img_file_names[i]), \
                os.path.join(new_path,img_file_names[i]))
    shutil.move(os.path.join(img_path, img_file_names[i][:-4]+".txt"), \
                os.path.join(new_path, img_file_names[i][:-4]+".txt"))