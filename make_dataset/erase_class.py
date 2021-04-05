import os
from tqdm import tqdm

#파일경로 가져오기
data_path = "/home/jngeun/darknet/training/dataset_fire"
file_names = os.listdir(data_path)

img_file_names = [file for file in file_names if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png")]

full_filename_img = []
for file_name in img_file_names:
    full_filename_img.append(os.path.join(data_path, file_name))

for img_file_name in tqdm(full_filename_img,desc= 'iterate list'):

    file_name = os.path.splitext(img_file_name)[0]
    txt_file_name = file_name + ".txt"

    lines_copy = []
    with open(txt_file_name,mode = 'r') as f:
        lines = f.readlines()
        lines_copy = lines

    with open(txt_file_name,mode = 'w') as f:
        for line in lines_copy:
            if int(line[0]) == 3:
                f.write(line)

    with open(txt_file_name, mode='r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            os.remove(img_file_name)
            os.remove(txt_file_name)