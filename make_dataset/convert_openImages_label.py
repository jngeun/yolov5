import os
import numpy as np
import cv2

#파일경로 가져오기
txt_path = "/home/jngeun/OIDv4_ToolKit/OID/Dataset/validation/Door handle/Label"
img_path = "/home/jngeun/OIDv4_ToolKit/OID/Dataset/validation/Door handle"
file_names = os.listdir(txt_path)

txt_file_names = [file for file in file_names if file.endswith(".txt")]

full_filename_txt = []
full_filename_img = []

for file_name in txt_file_names:
    full_filename_txt.append(os.path.join(txt_path, file_name))
    full_filename_img.append(os.path.join(img_path,file_name[:-4]+".jpg"))

for file_cnt in range(len(full_filename_txt)):
    label_list = np.loadtxt(full_filename_txt[file_cnt], dtype='str', delimiter=' ', ndmin=2)
    img = cv2.imread(full_filename_img[file_cnt])
    h,w = img.shape[:2]

    f = open(full_filename_txt[file_cnt], 'w')

    for i in range(len(label_list)):
        x_center = round((float(label_list[i][2])+float(label_list[i][4])) / (2*w),6)
        y_center = round((float(label_list[i][3])+float(label_list[i][5])) / (2*h),6)
        box_width = round((float(label_list[i][4]) - float(label_list[i][2])) / w,6)
        box_height = round((float(label_list[i][5]) - float(label_list[i][3])) / h,6)

        f.write(str(1))
        f.write(' ')
        f.write(str(x_center))
        f.write(' ')
        f.write(str(y_center))
        f.write(' ')
        f.write(str(box_width))
        f.write(' ')
        f.write(str(box_height))
        f.write('\n')
    f.close()