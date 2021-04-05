import requests
from pycocotools.coco import COCO
import cv2
from tqdm import tqdm

save_path = "/home/jngeun/dataset/temp/"

class_id = 'person'
classes_dict = {'person' : 0,'door':1,'handle' : 2,'refrigerator': 3}

#coco instance
coco = COCO('/home/jngeun/dataset/instances_train2017.json')
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]

catIds = coco.getCatIds(catNms=[class_id])
imgIds = coco.getImgIds(catIds=catIds )
images = coco.loadImgs(imgIds)


for im in tqdm(images[-600:-100], desc='iterate list'):

    #save image
    img_data = requests.get(im['coco_url']).content
    with open(save_path + im['file_name'], 'wb') as handler:
        handler.write(img_data)

    #save txt
    f = open(save_path + im['file_name'][:-4]+'.txt',mode= 'w')

    annIds = coco.getAnnIds(imgIds=im['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)

    img = cv2.imread(save_path + im['file_name'])
    h, w = img.shape[:2]

    for i in range(len(anns)):
        x_min = int(round(anns[i]['bbox'][0]))
        y_min = int(round(anns[i]['bbox'][1]))
        box_width = int(round(anns[i]['bbox'][2]))
        box_height = int(round(anns[i]['bbox'][3]))


        x_center = round((x_min + box_width / 2) / w, 6)
        y_center = round((y_min + box_height / 2) / h, 6)
        box_width = round(box_width / w,6)
        box_height = round(box_height / h,6)

        f.write(str(classes_dict[class_id]))
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