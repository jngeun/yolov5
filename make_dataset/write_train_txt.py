import os

#파일경로 가져오기
data_path = "/home/jngeun/darknet/training/dataset_fire/augmented_data"
txt_path = "/home/jngeun/darknet/training/model_disaster/temp.txt"
file_names = os.listdir(data_path)

img_file_names = [file for file in file_names if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png")]

full_filenames = []
for file_name in img_file_names:
    full_filenames.append(os.path.join(data_path, file_name))

with open(txt_path,mode='w') as f:
    for filename in full_filenames:
        f.write(filename)
        f.write('\n')