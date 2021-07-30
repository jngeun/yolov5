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
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPen
from PyQt5.QtCore import QTimer, QDateTime

import sys
import os
import numpy as np
from utils.general import clean_str
from utils.datasets import letterbox

#Bring .ui files & image
main_page = uic.loadUiType("MainWindow.ui")[0]
setting_page = uic.loadUiType("SettingWindow.ui")[0]

map_location = '/home/jngeun/yolov5/data/images/map.png'
battery_voltage_percent = 80

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
            print(f' success ({w}x{h} at {fps:.2f} FPS).')
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

        self.diff = []

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
        self.current_label = f""
        previous_label = f""

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
                    previous_label = self.current_label
                    self.current_label = f""
                    xx = f" "
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                        self.current_label += f"{n} {names[int(c)]}{'s' * (n > 1)} "

                    # Write results
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

                self.is_change = (self.current_label != previous_label)

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


class MainWindow(QMainWindow, main_page):
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

        #-------------------Map-----------------#
        map = QtGui.QPixmap(map_location)
        self.map.setPixmap(map)

        #--------------System Log---------------#
        self.systemLog.setPlainText('Start System.')

        #--------------System battery-----------#
        self.battery.setValue(battery_voltage_percent)
        #-----------------Clock-----------------#
        # creating a timer object
        timer = QTimer(self)
        # adding action to timer
        timer.timeout.connect(self.showDateTime)
        # update the timer every second
        timer.start(1000)
        self.clock.setFont(QtGui.QFont("궁서", 16))
        self.clock.setStyleSheet("Color : white")
        #-----------------Button----------------#
        self.btn1.setStyleSheet('image:url(data/images/btn1.png);border:0px;')
        self.btn2.setStyleSheet('image:url(data/images/btn2.png);border:0px;')
        self.btn3.setStyleSheet('image:url(data/images/btn3.png);border:0px;')

        self.setting.setStyleSheet('image:url(data/images/setting.png);border:0px;')
        self.setting.clicked.connect(self.settingPage)

        # Slider
        self.slider1.setValue(10)
        self.slider2.setValue(20)
        self.slider3.setValue(30)
        self.slider4.setValue(40)
        self.slider5.setValue(50)
        self.slider6.setValue(60)

        # Led
        self.cnt = 0
        self.led_up = cv2.imread('data/images/led_up.jpg', cv2.IMREAD_COLOR)
        self.led_down = cv2.imread('data/images/led_down.jpg', cv2.IMREAD_COLOR)
        self.led_left = cv2.imread('data/images/led_left.jpg', cv2.IMREAD_COLOR)
        self.led_right = cv2.imread('data/images/led_right.jpg', cv2.IMREAD_COLOR)

        self.led_width = self.led_up.shape[0]

    def threadEventHandler(self, im0):

        # webcam
        img = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape

        # label
        self.cnt =  (self.cnt + 1 ) % 100
        if self.cnt >= 0 and self.cnt < 25:
            self.led(img, 'U')
        elif self.cnt >= 25 and self.cnt <50 :
            self.led(img, 'R')
        elif self.cnt >=50 and self.cnt < 75:
            self.led(img, 'D')
        elif self.cnt >= 75:
            self.led(img, 'L')

        # pixmap
        w_scale,h_scale = 2, 2
        qImg = QtGui.QImage(img.data, w, h, w * c, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qImg)
        pixmap = pixmap.scaledToWidth(int(w_scale * w))
        pixmap = pixmap.scaledToHeight(int(h_scale * h))
        #self.drawLine(pixmap, QPoint(500, 400), self.label1.pos())
        self.video.setPixmap(pixmap)

        # System Log
        if self.th.is_change:
            self.systemLog.append(self.th.current_label + 'is detected')

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

    def drawLine(self,pixmap, p1, p2):
        painter = QtGui.QPainter(pixmap)
        painter.setPen(QPen(Qt.red, 10, Qt.SolidLine))
        painter.drawLine(p1, p2)
        painter.end
        self.update()

    def showDateTime(self):
        # getting current time
        current_time = QDateTime.currentDateTime()
        # converting QTime object to string
        label_time = current_time.toString('yyyy.MM.dd  hh:mm:ss')
        # showing it to the label
        self.clock.setText(label_time)

    def led(self, img, direction):
        if direction == 'U':
            img[0:self.led_width, 0:640] = cv2.add(img[0:self.led_width, 0:640],self.led_up)
        elif direction == 'R':
            img[0:480, 640 - self.led_width:640] = cv2.add(img[0:480,640 - self.led_width:640], self.led_right)
        elif direction == 'D':
            img[480 - self.led_width:480, 0:640] = cv2.add(img[480 - self.led_width :480, 0:640], self.led_down)
        elif direction == 'L' :
            img[0:480, 0: self.led_width] = cv2.add(img[0:480, 0: self.led_width], self.led_left)

        return img

    def settingPage(self):
        self.settingWindow = SettingPage()
        self.settingWindow.show()

class SettingPage(QMainWindow, setting_page):
    def __init__(self,parent=None):
        super().__init__(parent)

        self.setupUi(self)

if __name__ == '__main__':

    #Parsing Arguments in Terminal
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

    # QApplication : 프로그램을 실행시켜주는 클래스
    app = QtWidgets.QApplication(sys.argv)

    # UI 객체 생성
    MainWindow = MainWindow()

    # 프로그램 화면 보여줌
    MainWindow.show()

    # 프로그램을 이벤트루프로 진입
    app.exec_()
