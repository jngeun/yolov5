import os
import cv2
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tqdm import tqdm
from absl import flags
from absl.flags import FLAGS
import sys

FLAGS = flags.FLAGS

flags.DEFINE_string('data_path',None,'location of raw image')
flags.DEFINE_string('save_path',None,'location of augmented image')
flags.DEFINE_string('extension','jpg','file extension')
flags.DEFINE_integer('rotate',0,'write rotate amount using int')
flags.DEFINE_list('multiply',[1.0,1.0],'write brightness using list (ex)[0.5,0.5]')
flags.DEFINE_list('translate_px',[0,0],'write translate_px using list (ex)[10,10]')
flags.DEFINE_list('scale',[0.6,0.6],'write scale using list (ex)[0.5,0.5]')
flags.DEFINE_float('gaussian',0.,'write gaussianNoise parameter using int 0~0.2')

class Augmentation:
    def __init__(self):

        FLAGS(sys.argv)

        assert FLAGS.data_path != None, 'write the image_path'
        assert FLAGS.save_path != None, 'write the save_path'

        # 파일경로 가져오기
        data_path = FLAGS.data_path
        file_names = os.listdir(data_path)

        self.file_names = [file for file in file_names if file.endswith(FLAGS.extension)]

        self.numOfFile = len(self.file_names)

    def aug(self,filename):

        full_filename = os.path.join(FLAGS.data_path,filename)
        image = cv2.imread(full_filename)
        boxes = np.loadtxt(os.path.splitext(full_filename)[0] + ".txt", dtype=np.float, delimiter=' ', ndmin=2)
        h, w = image.shape[:2]

        ia_bounding_boxes = []
        box_class = []

        for box in boxes:
            ia_bounding_boxes.append(ia.BoundingBox(x1=w * (box[1] - box[3] / 2), y1=h * (box[2] - box[4] / 2),
                                                    x2=w * (box[1] + box[3] / 2), y2=h * (box[2] + box[4] / 2)))
            box_class.append(box[0])
        bbs = BoundingBoxesOnImage(ia_bounding_boxes, shape=image.shape)

        seq = iaa.Sequential([
            iaa.Multiply((FLAGS.multiply[0], FLAGS.multiply[1])),  # change brightness, doesn't affect BBs
            iaa.Affine(
                rotate=FLAGS.rotate,
                translate_px={"x": FLAGS.translate_px[0], "y": FLAGS.translate_px[1]},
                scale=(FLAGS.scale[0], FLAGS.scale[1])
            ),
            iaa.AdditiveGaussianNoise(scale=FLAGS.gaussian * 255)
        ])

        # Augment BBs and images.
        image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

        return image_aug, bbs_aug

    def save_file(self,img,box,filename):

        #save image

        image = cv2.imread(os.path.join(FLAGS.data_path,filename))
        h, w = image.shape[:2]

        image_file_name = os.path.splitext(filename)[0] + "_rotate" + str(FLAGS.rotate) + "_gaussianNoise" +\
                          str(FLAGS.gaussian) +".jpg"
        cv2.imwrite(os.path.join(FLAGS.save_path,image_file_name), img)

        #save label
        boxes = np.loadtxt(os.path.join(FLAGS.data_path,os.path.splitext(filename)[0] + ".txt"), dtype=np.float, delimiter=' ', ndmin=2)

        txt_file_name =os.path.join(FLAGS.save_path,os.path.splitext(image_file_name)[0] + ".txt")
        f = open(txt_file_name, 'w')
        for i in range(len(boxes)):
            after = box.bounding_boxes[i]

            x_center = round((after.x1 + after.x2) / (2 * w), 6)
            y_center = round((after.y1 + after.y2) / (2 * h), 6)
            box_width = round((after.x2 - after.x1) / w, 6)
            box_height = round((after.y2 - after.y1) / h, 6)

            f.write(str(int(boxes[i][0])))
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

if __name__ == '__main__':

    aug = Augmentation()

    for i,file in tqdm(enumerate(aug.file_names), desc='iterative list'):
        img, box = aug.aug(file)

        aug.save_file(img,box,file)
