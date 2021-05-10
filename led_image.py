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
    #cam_img = capture(CAM_ID)
    #h,w,c = cam_img.shape
    purple_grad = cv2.imread('data/images/purple_grad.png',cv2.IMREAD_COLOR)
    purple_grad = cv2.resize(purple_grad,(640,10),interpolation=cv2.INTER_CUBIC)
    h, w, c = purple_grad.shape

    gradation_img = np.zeros((h,w,c),dtype=np.uint8)

    # for x in range(0, w):
    #     gradation_img.itemset(50, x, 0, 255)
    #     gradation_img.itemset(50, x, 1, 255)
    #     gradation_img.itemset(50, x, 2, 255)

    cv2.imshow('gradation_img', purple_grad)
    cv2.imwrite('data/images/upper_led.jpg', purple_grad)
    #cv2.imshow('cam_img', cam_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()