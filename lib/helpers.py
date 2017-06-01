import numpy as np
import cv2
from collections import deque

class HysteresisSwitch:
  # Defines a state based switch which onset and offset are assymetric
  def __init__(self, low_t, high_t, lookup=1, inverse_bool=False, map_function=None):
    if inverse_bool :
      self.state = True
      self.past_values = deque([high_t for i in range(lookup)], lookup)
    else:
      self.state = False
      self.past_values = deque([low_t for i in range(lookup)], lookup)
    if map_function == None:
      self.map_function = self.local_avg
    else: 
      self.map_function = map_function

  def feed(self, value):
    self.past_values.append(value)
    coeff = self.map_function(list(value))
    if self.state and coeff < low_t:
      self.state = False 
    elif (not self.state) and coeff > high_t:
      self.state = True

  @staticmethod
  def local_avg(arr, weight=1):
    return sum(arr) * weight / len(arr)

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
