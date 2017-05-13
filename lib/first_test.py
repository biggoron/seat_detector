import argparse
import numpy as np
import imutils
import datetime
import time
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())

if args.get("video", None) is None:
  camera = cv2.VideoCapture(0)
  time.sleep(0.25)
else:
  camera = cv2.VideoCapture(args["video"])

firstframe = None

cv2.namedWindow("w1")
cv2.namedWindow("w2")
cv2.namedWindow("w3")
cv2.namedWindow("w4")
cv2.namedWindow("w5")
fgbg = cv2.createBackgroundSubtractorMOG2()
tracker = cv2.Tracker_create("KCF")
kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(45,45))
kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40))

if camera.isOpened():
  grabbed, firstframe = camera.read()
else:
  grabbed = False

while grabbed:
  grabbed, frame = camera.read()
  frame = imutils.resize(frame, width=500)
  imgs = [camera.read()[1] for i in range(5)]
  imgs = [imutils.resize(img, width=500) for img in imgs]
  grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in imgs]
  fgmasks = [fgbg.apply(gray) for gray in grays]
  fgmask = cv2.fastNlMeansDenoisingMulti(fgmasks, 2, 5, None, 4, 7, 15)
  output2 = fgmask
  fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)
  output3 = fgmask
  fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel1)
  output4 = fgmask
  fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, kernel2)
  fgmask = cv2.bilateralFilter(fgmask, 10, 75, 75)
  output5 = fgmask
  contours = cv2.findContours(fgmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
  rects = [cv2.boundingRect(cnt) for cnt in contours]
  for r in rects:
    x, y, w, h = r
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
  #cv2.drawContours(frame, contours, -1, (0, 255, 0), 3)
  cv2.imshow("w1", frame)
  cv2.imshow("w2", output2)
  cv2.imshow("w3", output3)
  cv2.imshow("w4", output4)
  cv2.imshow("w5", output5)
  key = cv2.waitKey(20)
  if key == 27:
    break
cv2.destroyAllWindows("preview")
camera.release()
