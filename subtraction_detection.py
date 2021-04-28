import cv2
import numpy as np

vs = cv2.VideoCapture("../data/punto_padel.avi")

ret, f1 = vs.read()
ret, f2 = vs.read()

while True:
    diff = cv2.absdiff(f1, f2)
    hsv = cv2.cvtColor(diff, cv2.COLOR_BGR2HSV)
    blur = cv2.GaussianBlur(hsv, (5, 5), 0)
    mask = cv2.inRange(blur, (36, 25, 25), (70, 255, 255))
    imask = mask > 0
    green = np.zeros_like(blur, np.uint8)
    green[imask] = blur[imask]

    _, thresh = cv2.threshold(green, 220, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    contours, _ = cv2.findContours(cv2.cvtColor(dilated, cv2.COLOR_BGR2GRAY), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cv2.imshow("diff", diff)
    cv2.imshow("hsv", hsv)
    cv2.imshow("blur", blur)
    cv2.imshow("mask", green)
    cv2.imshow("th", thresh)
    cv2.imshow("dilated", dilated)

    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)

        cv2.rectangle(f1, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("feed", f1)

    f1 = f2
    ret, f2 = vs.read()
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
vs.release()