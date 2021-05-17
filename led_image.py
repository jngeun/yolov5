import cv2
import numpy as np

CAM_ID = 0
LED_WIDTH = 10
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

    rampl = (np.linspace(1, 0, LED_WIDTH) * 255).astype(np.uint8)
    rampl = np.tile(np.transpose(rampl), (480, 1))
    rampl = cv2.merge([rampl, rampl, rampl])
    cv2.imwrite('data/images/led_left.jpg', rampl)
    rampr = cv2.rotate(rampl,cv2.ROTATE_180)
    cv2.imwrite('data/images/led_right.jpg', rampr)

    rampu = (np.linspace(1, 0, LED_WIDTH) * 255).astype(np.uint8)
    rampu = np.tile(np.transpose(rampu), (640, 1))
    rampu = cv2.merge([rampu, rampu, rampu])
    rampu = cv2.rotate(rampu, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite('data/images/led_up.jpg',rampu)
    rampd = cv2.rotate(rampu, cv2.ROTATE_180)
    cv2.imwrite('data/images/led_down.jpg',rampd)



    # roi = cam_img[0:30,0:640]
    #
    # dst = cv2.add(roi,up)
    # cam_img[0:30, 0:640] = dst

    cv2.imshow('cam_img', rampl)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # purple_grad = cv2.imread('data/images/purple_grad.png',cv2.IMREAD_COLOR)
    # purple_grad = cv2.resize(purple_grad,(640,10),interpolation=cv2.INTER_CUBIC)
    #cv2.imwrite('data/images/upper_led.jpg', purple_grad)