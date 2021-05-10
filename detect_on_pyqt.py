import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_label
from utils.torch_utils import select_device, load_classifier, time_synchronized

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QWidget
from PyQt5 import QtGui
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtCore
from PyQt5 import uic
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen

import sys
import os
import numpy as np
from utils.general import clean_str
from utils.datasets import letterbox

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=640, stride=32):
        self.mode = 'stream'
        self.img_size = img_size
        self.stride = stride

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().strip().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = [clean_str(x) for x in sources]  # clean source names for later
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print(f'{i + 1}/{n}: {s}... ', end='')
            self.cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert self.cap.isOpened(), f'Failed to open {s}'
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = self.cap.read()  # guarantee first frame
            #thread = Thread(target=self.update, args=([i, self.cap]), daemon=True)
            print(f' success ({w}x{h} at {fps:.2f} FPS).')
            #thread.start()
        print('')  # newline

        # check for common shapes
        s = np.stack([letterbox(x, self.img_size, stride=self.stride)[0].shape for x in self.imgs], 0)  # shapes
        self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        self.cap.grab()
        success, im = self.cap.retrieve()
        self.imgs[0] = im if success else self.imgs[0] * 0

        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        # Letterbox
        img = [letterbox(x, self.img_size, auto=self.rect, stride=self.stride)[0] for x in img0]

        # Stack
        img = np.stack(img, 0)

        # Convert
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years

class Detect(QtCore.QThread):
    # 데이터 전달 시 타입을 명시
    threadEvent = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, parent = None):
        super().__init__()
        self.main = parent
        self.img = None
        self.label = []

    def run(self,save_img=False):

        source, weights, view_img, save_txt, imgsz = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(
                device).eval()

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            #view_img = check_imshow()
            view_img = True
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)

        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        t0 = time.time()
        for path, img, im0s, vid_cap in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = time_synchronized()
            pred = model(img, augment=opt.augment)[0]

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)
            t2 = time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            # Process detections
            for i, det in enumerate(pred):  # detections per image
                if webcam:  # batch_size >= 1
                    p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                else:
                    p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                p = Path(p)  # to Path
                save_path = str(save_dir / p.name)  # img.jpg
                txt_path = str(save_dir / 'labels' / p.stem) + (
                    '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                s += '%gx%g ' % img.shape[2:]  # print string
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    self.label = []
                    self.location = []
                    for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                -1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            #label = f'{names[int(cls)]} {conf:.2f}'
                            label = f'{names[int(cls)]}'
                            plot_one_label(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

                # Print time (inference + NMS)
                print(f'{s}Done. ({t2 - t1:.3f}s)')

                # Stream results
                if view_img:
                    self.img = im0
                    self.threadEvent.emit(self.img)
                    #cv2.imshow(str(p), im0)
                    cv2.waitKey(1)  # 1 millisecond

        if save_txt or save_img:
            s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            print(f"Results saved to {save_dir}{s}")

        print(f'Done. ({time.time() - t0:.3f}s)')

form_class = uic.loadUiType("MyWindow.ui")[0]

class MyWindow(QMainWindow, form_class):
    def __init__(self,parent=None):
        super().__init__(parent)

        #GUI 배치
        self.setupUi(self)
        self.initializeUi()

        #쓰레드 인스턴스 생성
        self.th = Detect(self)
        # 쓰레드 이벤트 연결
        self.th.threadEvent.connect(self.threadEventHandler)
        #쓰레드 시작
        print(f'Thread start')
        self.th.start()




    def initializeUi(self):

        #label
        # self.label = [self.label1, self.label2, self.label3, self.label4, self.label5]
        # self.label_center = []
        # for i,label in enumerate(self.label):
        #     x_center = (label.x() + label.width())
        #     y_center = (label.y() + label.height())
        #     self.label_center.append(QPoint(x_center,y_center))

        # #led
        # self.led_orange = (241, 162,20)
        # self.led_blue = (22, 179, 251)
        # self.led_gray = (59, 59, 59)
        # self.pyqt_led = [self.led_front, self.led_right, self.led_back, self.led_left]
        # self.led_toggle = 0
        #
        # for i in range(4):
        #     self.ledOff(i)

        #map
        map = cv2.imread('/home/jngeun/yolov5/data/map.png', cv2.COLOR_BGR2RGB)
        h, w, c = map.shape
        map = QtGui.QImage(map.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        map = QtGui.QPixmap.fromImage(map)
        self.map.setPixmap(map)

        #system log
        self.systemLog.setPlainText('시스템을 시작합니다.')


    def threadEventHandler(self, im0):

        #webcam
        qImg = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        h, w, c = qImg.shape
        w_scale,h_scale = 1.8, 1.8
        qImg = QtGui.QImage(qImg.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        pixmap = pixmap.scaledToWidth(int(w_scale * w))
        pixmap = pixmap.scaledToHeight(int(h_scale * h))
        #self.drawLine(pixmap, QPoint(500, 400), self.label1.pos())
        self.video.setPixmap(pixmap)

        #system Log
        self.systemLog.append('사람을 발견했습니다')

        #label
        # for i,label in enumerate(self.th.label):
        #     pixmap = QtGui.QPixmap(self.label[i].size())
        #     pixmap.fill(Qt.transparent)
        #     qp = QPainter(pixmap)
        #     pen = QPen(Qt.red, 3)
        #     qp.setPen(pen)
        #     qp.drawLine(QPoint(500,400),self.label_center[i])
        #     qp.drawText(pixmap.rect(), Qt.AlignCenter, label)
        #     qp.end()
        #     self.label[i].setPixmap(pixmap)


        #led
        '''
        self.led_toggle = ( self.led_toggle + 1 ) % 40

        if self.led_toggle == 0:
            self.ledOn(0)
            self.ledOff(3)
        elif self.led_toggle == 10:
            self.ledOn(1)
            self.ledOff(0)
        elif self.led_toggle == 20:
            self.ledOn(2)
            self.ledOff(1)
        elif self.led_toggle ==30:
            self.ledOn(3)
            self.ledOff(2)
        '''


    def drawLine(self,pixmap, p1, p2):
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QPen(Qt.red, 10, Qt.SolidLine))
        painter.drawLine(p1, p2)
        painter.end
        self.update()

    # def ledOn(self, direction):
    #     self.pyqt_led[direction].setStyleSheet(f'background-color:rgb{self.led_orange}')
    #
    # def ledOff(self,direction):
    #     self.pyqt_led[direction].setStyleSheet(f'background-color:rgb{self.led_gray}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov5s.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='data/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    check_requirements(exclude=('pycocotools', 'thop'))

    #detection label
    detection_label = []

    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QtWidgets.QApplication(sys.argv)
    # 객체 생성
    myWindow = MyWindow()
    # 프로그램 화면 보여줌
    myWindow.show()
    # 프로그램을 이벤트루프로 진입
    app.exec_()
