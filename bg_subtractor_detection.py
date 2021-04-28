import cv2
import numpy as np

# Templates preparation
ball_img = "../data/balls/ball_{idx}.PNG"
images = []
for i in range(2, 3):
    image = cv2.imread(ball_img.format(idx=i), 0)
    w, h = image.shape[::-1]
    images.append({"image": image, "w": w, "h": h})
# Video preparation
vs = cv2.VideoCapture("../data/punto_padel.avi")

object_detector = cv2.createBackgroundSubtractorMOG2(history=150, varThreshold=25)

while True:
    ret, frame = vs.read()

    # 1. Object Detection
    mask = object_detector.apply(frame)
    _, mask = cv2.threshold(mask, 254, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # for image in images:
    #     res = cv2.matchTemplate(gray, image["image"], cv2.TM_CCOEFF_NORMED)
    #     threshold = 0.90
    #     loc = np.where(res >= threshold)
    #
    #     for pattern in zip(*loc[::-1]):
    #         cv2.rectangle(frame, pattern, (pattern[0]+image["w"], pattern[1]+image["h"]), (255, 0, 0), 2)
    #         print("detected pattern")

    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:

        area = cv2.contourArea(contour)

        if area > 100:
            #cv2.drawContours(frame, [contour], -1, (0, 255, 0), 1)
            x, y, w, h = cv2.boundingRect(contour)
            cv2.circle(frame, (int(x+w/2), int(y+h/2)), 7, (0, 255, 0), 2)

    cv2.imshow("main", frame)
    cv2.imshow("mask", mask)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
vs.release()