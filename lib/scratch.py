import argparse
import numpy as np
import imutils
import datetime
import time
import cv2

class PrintMng:
  # Handles the printing windows
  
  def __init__(self):
    # Windows list
    self.windows = []
  
  def display(self, img, title):
    # Uses or create window to display image
    if not (title in self.windows):
      self.windows.append(title)
      cv2.namedWindow(title)
    cv2.imshow(title, img)
  
  def close_all(self):
    # At the end of the experiment
    cv2.destroyAllWindows("preview")

class Box:
  def __init__(self, x, y, w, h):
    self.w = w
    self.h = h
    self.x = x
    self.y = y
  def shared_area(self, box):
    a = max(self.x, box.x)
    b = min(self.x + self.w, box.x + box.w)
    if a >= b:
      return 0
    c = min(self.y, box.y)
    d = max(self.y + self.h, box.y + box.h)
    if d >= c:
      return 0
    return (b-a)*(c-d)
  def dist(self, box):
    dx = abs((self.x + self.w / 2) - (box.x + box.w / 2))
    dy = abs((self.y + self.h / 2) - (box.y + box.h / 2))
    return dx*dx + dy*dy
  def equal(self, box):
    dx = abs((self.x + self.w / 2) - (box.x + box.w / 2))
    dy = abs((self.y + self.h / 2) - (box.y + box.h / 2))
    dw = abs(self.w - box.w)
    dh = abs(self.h - box.h)
    d = dx*dx + dy*dy + dw*dw + dh*dh
    return True if d < (self.w*self.w + self.h*self.h) / 4 else False
    
  def coord(self):
    return self.x, self.y, self.w, self.h

class BoxMem:
  def __init__(self, max_size=300, min_size=150, init_w=3, del_lim=0, max_w=5, decay=0.2):
    self.init_w = init_w
    self.del_lim = del_lim
    self.max_size = max_size
    self.min_size = min_size
    self.max_w = max_w
    self.decay = decay
    self.boxes = []

  @staticmethod
  def box_size(box, M, m, R=1.5):
    l = box.w + box.h
    if l > M : # Too big
      return 1
    r = box.h / box.w
    total = l * (1 - 0.6 * abs(R-r))
    if total < m :  # Too small or strange ratio
      return -1
    return 0

  def feed(self, box_list, debug_frame=None):
    for b in box_list:
      size = BoxMem.box_size(b, self.max_size, self.min_size)
      if size == 0:
        cv2.rectangle(debug_frame, (b.x, b.y), (b.x+b.w, b.y+b.h), (255, 0, 0), 2)
        candidates = [[c, b.dist(c[0])] for c in self.boxes if b.equal(c[0])]
        if len(candidates) > 0:
          c = min(candidates, key=lambda x:x[1])
          i = self.boxes.index(c[0])
          self.boxes[i][0].x = (b.x + c[0][0].x) // 2
          self.boxes[i][0].y = (b.y + c[0][0].y) // 2
          self.boxes[i][0].w = (b.w + c[0][0].w) // 2
          self.boxes[i][0].h = (b.h + c[0][0].h) // 2
          self.boxes[i][1] = 5
        else:
          self.boxes.append([b, self.init_w])
      if size == -1:
        cv2.rectangle(debug_frame, (b.x, b.y), (b.x+b.w, b.y+b.h), (0, 255, 0), 2)
        for i in range(len(self.boxes)):
          area = b.shared_area(self.boxes[i][0])
          s = self.boxes[i][0].w * self.boxes[i][0].h  
          self.boxes[i][1] += 24 * area / s
      if size == 1:
        cv2.rectangle(debug_frame, (b.x, b.y), (b.x+b.w, b.y+b.h), (0, 0, 255), 2)
        candidates = [c for c in self.boxes if b.shared_area(c[0]) > 0.7 * c[0].w * c[0].h]
        for c in candidates:
          i = self.boxes.index(c)
          self.boxes[i][1] = 5
    for b in self.boxes:
      b[1] -= self.decay
      if b[1] > self.max_w:
        b[1] = max_w
      if b[1] < self.del_lim:
        i = self.boxes.index(b)
        self.boxes.pop(i)
  def get(self):
    return [b[0] for b in self.boxes if b[1] > self.del_lim + 1]

def blurr(img_set, fgbg, blur_kernel, denoise_kernel):
  # Sequence of processing to remove useless information on the picture

  # Removes colors
  grays = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_set]
  # Removes background, put forground in white
  fgmasks = [fgbg.apply(gray) for gray in grays]
  # Denoises the fgbg picture
  fgmask = cv2.fastNlMeansDenoisingMulti(fgmasks, 2, 5, None, 4, 7, 15)
  # Blurr the found foreground to connect close patches corresponding to the same object
  blurred = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, blur_kernel)
  # Removes small independent moving objects.
  denoised = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, denoise_kernel)
  # Blurr again
  blurred = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, blur_kernel)
  # Smooth out the image, without altering edges.
  cleaned = cv2.bilateralFilter(blurred, 10, 75, 75)
  return cleaned

def bound(img):
  # delimit patches in the image, bound them by rectangles
  contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
  rects = [cv2.boundingRect(cnt) for cnt in contours]
  return rects

# Program initialization and main processing loop
def main(img_size=500):

  # Handles the printing windows
  printer = PrintMng()

  # Choose between live or video
  ap = argparse.ArgumentParser()
  ap.add_argument("-v", "--video", help="path to the video file")
  #ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
  args = vars(ap.parse_args())

  # Initialize camera
  if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
  else:
    camera = cv2.VideoCapture(args["video"])

  # Initialize elements needed for the precessing of the stream
  fgbg = cv2.createBackgroundSubtractorMOG2()
  #tracker = cv2.Tracker_create("KCF")
  kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
  kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40))
  boxes = BoxMem()

  # Initialize variable describing camera state
  if camera.isOpened():
    grabbed, firstframe = camera.read()
  else:
    grabbed = False

  # Until the last image of the video stream
  while grabbed:
    # Take current picture and resize
    grabbed, frame = camera.read()
    frame = imutils.resize(frame, width=img_size)
    # Take set of last 5 pictures and resize
    imgs = [camera.read()[1] for i in range(5)]
    imgs = [imutils.resize(img, width=img_size) for img in imgs]
    # Remove useless details and colors
    shape_approx = blurr(imgs, fgbg, kernel2, kernel1)
    # Bound the objects in the pictures by boxes
    local_rects = bound(shape_approx)
    # Print the boxes on the current frame
    box_list = [Box(rect[0], rect[1], rect[2], rect[3]) for rect in local_rects]
    boxes.feed(box_list, debug_frame = frame)
    box_list = boxes.get()
    for b in box_list:
      x, y, w, h = b.coord()
      cv2.rectangle(frame, (x-5, y-5), (x+w+5, y+h+5), (255, 255, 255), 2)
    # Display the frame
    printer.display(frame, "w1")
    # Manual break
    key = cv2.waitKey(20)
    if key == 27:
      break
  # Close resources
  printer.close_all
  camera.release()

main()
