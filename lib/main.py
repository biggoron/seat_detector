import argparse
import numpy as np
import human_recognizer as hr
import imutils
import datetime
import time
import cv2
from helpers import *
from box import *
from image_helpers import *
from seat import *

# Program initialization and main processing loop
def main(
  img_size=500, # Resizes the video. The given number is the target width.
  max_size=400, # Max dimension of humans. The given number is width + height.
  min_size=150, # Min dimension of humans. The given number is width + height.
):

  # ------ Choose between live or video file -----------
  ap = argparse.ArgumentParser()
  ap.add_argument("-v", "--video", help="path to the video file")
  #ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
  args = vars(ap.parse_args())
  # ------------------------------------------------------

  # Initialize camera with the proper video source
  if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(0.25)
  else:
    camera = cv2.VideoCapture(args["video"])
  #------------------------------------------------

  # -----Initialize elements needed for the precessing of the video stream---

  # Handles the printing of windows
  printer = PrintMng()

  # DNN Human Recognizer
  recognizer = hr.HumanRecognizer("model/graph_zoom.pb", "model/labels_zoom.txt")

  # Background substractor
  fgbg = cv2.createBackgroundSubtractorMOG2()

  # Smoothing kernels
  kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
  kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(40,40))

  # Memory for tracking the boxes containing moving objects
  boxes = BoxMem(recognizer, max_size=max_size, min_size=min_size)

  # Memory for tracking the seats
  seats = SeatMem(seat_nb = 2, decay = 0.1) # seat_nb is the expected number of seats
  seat_list = []

  # Initialize the time counter for seat long memory update
  last_access = None
  # ------------------------------------------------------

  # ---- Initialize variables describing camera state -----
  if camera.isOpened():
    grabbed, firstframe = camera.read()
  else:
    grabbed = False
  # ------------------------------------------------------

  # looping until the last image of the video stream
  while grabbed:
    # Take current frame and resize it
    grabbed, frame = camera.read()
    frame = imutils.resize(frame, width=img_size)

    # Remove useless details and colors
    shape_approx = blur(frame, fgbg, kernel2, kernel1)

    # Bound the objects in the pictures by boxes
    local_rects = bound(shape_approx)

    # Print the box coordinates to a Box object (box.py)
    box_list = [Box(rect[0], rect[1], rect[2], rect[3]) for rect in local_rects]
    # Updates the box memory with the boxes found in the current frame.
    boxes.feed(box_list, frame)
    # Get the list of box that could contain humans
    box_list = boxes.get()

    # USED FOR DEBUGGING
    # Print the boxes on the current frame for debugging
    for b in box_list:
      x, y, w, h = b[0].coord()
      cropped = frame[x:x+w, y:y+h]
      #human = recognizer.is_human(image=cropped)
      color = (255, 255, 255)
      cv2.rectangle(frame, (x-15, y-15), (x+w+15, y+h+15), color, 2)

    # Updates the potential seats with the detected boxes.
    seats.feed(box_list)

    # Updates the detected seats with the potential seats.
    # mem_span is the time seated after which the seating place is considered a seat.
    mem_span = 45 #seconds
    if last_access == None:
      last_access = time.time()
      
    if ((time.time() - last_access) > (mem_span / 3)):
      seats.short_to_long_mem(trigger_on = mem_span)
      # Get the most probable seats
      # NOTICE: should be run only when one wants to see the seat list
      seat_list = seats.get()
        
      # TODO apply DNN to check the seats
      # maybe if there is someone in a seat declare it as a box and blacken the corresponding area after BG extraction.

      # memorize the time
      last_access = time.time()

    # USED FOR DEBUGGING
    # prints in yellow the seats

    for s in seat_list:
      x, y, w, h = s[1].coord()
      cropped = frame[x:x+w, y:y+h]
      #if recognizer.is_human(image=cropped):
      color = (0, 255, 255)
      #else:
      #color = (255, 0, 255)
      cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    # DEBUG
    # Display the frame
    printer.display(frame, "w1")

    # Manual break
    key = cv2.waitKey(20)
    if key == 27:
      break

  # Close resources
  printer.close_all
  recognizer.close()
  camera.release()

main(max_size=370, min_size=180)
