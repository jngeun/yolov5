import os
import sys

#파일경로 가져오기
data_path = "/home/jngeun/darknet/training/dataset_door"
file_names = os.listdir(data_path)

txt_file_names = [file for file in file_names if file.endswith(".txt")]

full_filename_txt = []
for file_name in txt_file_names:
    full_filename_txt.append(os.path.join(data_path, file_name))

#erase empty txt file and same name img
for file_name in full_filename_txt:
    with open(file_name,mode = 'r') as f:
        lines = f.readlines()
        if len(lines) == 0:
            #erase txt file
            os.remove(file_name)
            print("remove "+ file_name)
            #erase img file
            img_file_name = os.path.splitext(file_name)[0]
            isJpg = os.path.isfile(img_file_name+".jpg")
            isJpeg = os.path.isfile(img_file_name + ".jpeg")
            isPng = os.path.isfile(img_file_name + ".png")
            if isJpg:
                img_file_name = img_file_name + ".jpg"
            elif isJpeg:
                img_file_name = img_file_name + ".jpeg"
            elif isPng:
                img_file_name = img_file_name + ".png"
            else:
                print("no file extension :", img_file_name)
                sys.exit()

            os.remove(img_file_name)
            print("remove " + img_file_name)