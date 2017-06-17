import time
import cv2

class Box:
  # Represents a rectangle on the screen
  def __init__(self, x, y, w, h):
    # position, width and height
    self.w = w
    self.h = h
    self.x = x
    self.y = y
  def shared_area(self, box):
    # Returns the superficy of the area covered by both
    # self and box.
    a = max(self.x, box.x)
    b = min(self.x + self.w, box.x + box.w)
    if a >= b:
      return 0
    c = min(self.y, box.y)
    d = max(self.y + self.h, box.y + box.h)
    if d >= c:
      return 0
    return (b-a)*(c-d)
  def area(self):
    return w*w + h*h
  def dist(self, box):
    # distance between the centers of self and box
    dx = abs((self.x + self.w / 2) - (box.x + box.w / 2))
    dy = abs((self.y + self.h / 2) - (box.y + box.h / 2))
    return dx*dx + dy*dy
  def equal(self, box):
    # Boolean. Tells if self and Box can be considered to be the same.
    # TODO: introduce a coefficient instead of the "/4" at the end.
    dx = abs((self.x + self.w / 2) - (box.x + box.w / 2))
    dy = abs((self.y + self.h / 2) - (box.y + box.h / 2))
    dw = abs(self.w - box.w)
    dh = abs(self.h - box.h)
    d = dx*dx + dy*dy + dw*dw + dh*dh
    return True if d < (self.w*self.w + self.h*self.h) / 4 else False

  @staticmethod
  def avg(box_list):
    # Average box for a list of boxes
    x_list = [box.x for box in box_list]
    y_list = [box.y for box in box_list]
    w_list = [box.w for box in box_list]
    h_list = [box.h for box in box_list]
    x_avg = sum(x_list) // len(x_list)
    y_avg = sum(y_list) // len(y_list)
    w_avg = sum(w_list) // len(w_list)
    h_avg = sum(h_list) // len(h_list)
    return Box(x_avg, y_avg, w_avg, h_avg)
    
  def coord(self):
    # position, width and height
    return self.x, self.y, self.w, self.h

class BoxMem:
  # Keeps track of boxes which are the size of a human.
  # Boxes have weights to keep track of their likeliness
  # of being a human.
  def __init__(self, recognizer, max_size=650, min_size=250, init_w=3, del_lim=0, max_w=6, decay=0.5, max_id=50):
    self.init_w = init_w # Initial weight
    self.del_lim = del_lim # delete boxes with weight below limit.
    self.max_size = max_size # max size of a human (width + height)
    self.min_size = min_size  # min size of a human (width + height)
    self.recognizer = recognizer # DNN model object to check humans
    self.max_w = max_w # Maximum weight
    self.decay = decay # Weight decrement applied at each frame (to forget boxes where nothing happens)
    self.ids = [False] * max_id # Ids of the boxes
    self.boxes = [] # list of the boxes that have the good size and enough activity within

  @staticmethod
  def box_size(box, M, m, R=1.5):
    # Decides if the box is too small or too big
    l = box.w + box.h
    if l > M : # Too big
      return 1
    r = box.h / box.w
    # TODO: replace the 0.6 by a coefficient
    # to ponder the importance of the w/h ratio
    total = l * (1 - 0.6 * abs(R-r))
    if total < m :  # Too small or strange ratio
      return -1
    return 0

  def box_id(self):
    # Generates and keeps track of box Ids
    try_id = 0
    while self.ids[try_id]:
      try_id += 1
    self.ids[try_id] = True
    return try_id

  def remove_box(self, i):
    # deletes box and untracks associated id
    box_i = self.boxes[i][2]
    self.boxes.pop(i)
    self.ids[box_i] = False

  def feed(self, box_list, frame):
    # Updates the good box list with the box detected at the current frame
    for b in box_list:
      size = BoxMem.box_size(b, self.max_size, self.min_size)
      if size == -1:
        # Box is to small or with strange ratio
        sub_im = frame[b.y : (b.y+b.h), b.x:b.x+b.w]
        #if self.recognizer.is_human(image=sub_im):
        #size == 1
        #self.min_size = (self.min_size*2 + (b.w + b.h)*0.8) / 3
        #cv2.rectangle(frame, (b.x+3, b.y+3), (b.x+b.w-3, b.y+b.h-3), (57, 178, 0), 2)
        #else:
        for i in range(len(self.boxes)):
          # Reinforce learned good boxes if they have common area with the small box
          area = b.shared_area(self.boxes[i][0])
          s = self.boxes[i][0].w * self.boxes[i][0].h  
          self.boxes[i][1] += 12 * area / s
        # DEBUG
        # Draw it in green on the screen
        # cv2.rectangle(frame, (b.x, b.y), (b.x+b.w, b.y+b.h), (160, 255, 108), 2)
      if size == 0:
        # The detected box has the good size
        # DEBUG
        # Print it in blue
        # cv2.rectangle(frame, (b.x, b.y), (b.x+b.w, b.y+b.h), (34, 25, 176), 2)
        # Check if there are learned boxes very close to the detected box
        candidates = [[c, b.dist(c[0])] for c in self.boxes if b.equal(c[0])]
        if len(candidates) > 0:
          # If so, take the closest
          c = min(candidates, key=lambda x:x[1])
          i = self.boxes.index(c[0])
          # move it to match the currently detected box
          self.boxes[i][0].x = (b.x + c[0][0].x) // 2
          self.boxes[i][0].y = (b.y + c[0][0].y) // 2
          self.boxes[i][0].w = (b.w + c[0][0].w) // 2
          self.boxes[i][0].h = (b.h + c[0][0].h) // 2
          # Put high confidence (high weight) on the box
          self.boxes[i][1] = 5
        else:
          # Create a new learned box, based on the detected box
          self.boxes.append([b, self.init_w, self.box_id()])
      if size == 1:
        # The detected box is too big.
        # DEBUG
        # draw it in red
        # cv2.rectangle(frame, (b.x, b.y), (b.x+b.w, b.y+b.h), (0, 0, 255), 2)
        # Check is there are learned boxes within
        candidates = [c for c in self.boxes if b.shared_area(c[0]) > 0.5 * c[0].w * c[0].h]
        for c in candidates:
          # reinforce the weight of those learned boxes
          i = self.boxes.index(c)
          self.boxes[i][1] = 5
    for b in self.boxes:
      sub_im = frame[b[0].y : (b[0].y+b[0].h), b[0].x:b[0].x+b[0].w]
      if ((b[1] < self.del_lim + 2*self.decay) and self.recognizer.is_human(image=sub_im)):
        b[1] = self.max_w
      else:
        # Forget a little each learned box
        b[1] -= self.decay
        # If the weight is too high, set it to the max
        if b[1] > self.max_w:
          b[1] = self.max_w
        if b[1] < self.del_lim:
          # If the weight is too small, delete the learned box
          # TODO: should check if there is human in it before throwing it
          i = self.boxes.index(b)
          self.remove_box(i)

  def get(self):
    # returns the learned boxes
    return [[b[0], b[2]] for b in self.boxes]
