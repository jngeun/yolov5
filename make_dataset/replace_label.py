import os
import numpy as np

#파일경로 가져오기
data_path = "/home/jngeun/darknet/training/dataset_fire"
file_names = os.listdir(data_path)

txt_file_names = [file for file in file_names if file.endswith(".txt")]

full_filename_txt = []
for file_name in txt_file_names:
    full_filename_txt.append(os.path.join(data_path, file_name))

for file_cnt in range(len(full_filename_txt)):
    print(full_filename_txt[file_cnt])
    label_list = np.loadtxt(full_filename_txt[file_cnt], dtype='str', delimiter=' ', ndmin=2)

    f = open(full_filename_txt[file_cnt], 'w')

    for i in range(len(label_list)):
        if int(label_list[i][0]) == 0:
            label_list[i][0] = 4
        elif int(label_list[i][0]) == 1:
            label_list[i][0] = 3
        else:
            print(full_filename_txt[file_cnt], " : index is weird")

        f.write(str(label_list[i][0]))
        f.write(' ')
        f.write(label_list[i][1])
        f.write(' ')
        f.write(label_list[i][2])
        f.write(' ')
        f.write(label_list[i][3])
        f.write(' ')
        f.write(label_list[i][4])
        f.write('\n')

    f.close()


