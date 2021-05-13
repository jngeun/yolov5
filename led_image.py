import cv2
import numpy as np

CAM_ID = 0

def capture(camid=CAM_ID):
    cam = cv2.VideoCapture(camid)
    if cam.isOpened() == False:
        print('Cant open the cam (%d)' % camid)
        return None

    ret, frame = cam.read()
    if frame is None:
        print('frame is not exist')
        return None

    cam.release()
    return frame

if __name__ == "__main__":
    cam_img = capture(CAM_ID)
    h,w,c = cam_img.shape
    # upper_led = cv2.imread('data/images/upper_led.jpg',cv2.IMREAD_COLOR)

    rampl = (np.linspace(1, 0, 30) * 255).astype(np.uint8)
    rampl = np.tile(np.transpose(rampl), (480, 1))
    rampl = cv2.merge([rampl, rampl, rampl])
    up = cv2.rotate(rampl,cv2.ROTATE_180)
    cv2.imwrite('data/images/led_right.jpg',up)

    roi = cam_img[0:30,0:640]

    dst = cv2.add(roi,up)
    cam_img[0:30, 0:640] = dst

    cv2.imshow('cam_img', cam_img)
    cv2.imshow('-',up)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # purple_grad = cv2.imread('data/images/purple_grad.png',cv2.IMREAD_COLOR)
    # purple_grad = cv2.resize(purple_grad,(640,10),interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite('data/images/upper_led.jpg', purple_grad)