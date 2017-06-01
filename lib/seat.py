from box import *
import numpy as np
import time

class SeatMem:
  # This class holds the information about potential seats
  # (boxes that stand still for long enough), and the
  # information about detected seats (potential seats
  # that occur often at the same place).
  def __init__(self,
              seat_nb, # Number of expected seats on the view
              init_w = 0.2, # Initial weight for a seat
              max_w = 100, # Maximum weight for a seat
              max_box_id=50, # Initialize the maximum number of simultaneous boxes coming in. Adaptative.
              max_seat_nb=100, # Initialize the maximum number of seat ids. Adaptative
              decay = 0.1): # Decrements the weight of a seat if no-one is on the seat.
    self.init_w = init_w 
    self.max_w = max_w
    self.seat_nb = seat_nb
    self.short_mem = {} # Stores sequences of positions for detected boxes.
    self.long_mem = {} # If a sequence in short_mem is stable, it goes into long_mem and becomes a seat.
    self.decay = decay
    self.seat_ids = [None] * max_seat_nb # Registers the ids of the seats.
    self.box_id = max_box_id # TODO: set it to the same value than the one of Boxmem

  def feed(self, box_list):
    # Iterate over the boxes detected in the frame
    for box in box_list:
      # Separate the information.
      box_id = box[1]
      box = box[0]
      if not box_id in self.short_mem:
        # The id of the box is unknown, new object to track.
        self.short_mem[box_id] = []
      # Each index of short_mem has a list of position of an object.
      # Append the new position and the time at the end of the list of positions.
      self.short_mem[box_id].append([box, time.time()])

  def short_to_long_mem(self,
    trigger_on = 10, # The time needed for someone to stay still before being
                     # considered "seated"
    static_lim = 100 # The authorized movement margin for someone who is seated.
  ):
    #  Once in a while the system needs to throw moving trajectories in
    # short_mem, and use the static ones to reinforce its knowledge about
    # seats.

    # Items we should stop tracking. (No news in a while from them).
    pop_list_short = []

    # k is the id of the tracked object (represented by a box),
    # v is the trajectory associated with it
    for k, v in self.short_mem.items():
      # positions in the trajectory that are too old and should be
      # deleted
      pop_list = []

      last_time = time.time()
      for i in range(len(v)):
        if last_time - v[i][1] > trigger_on:
          # If the position is too old list it to be deleted
          pop_list.append(i)
      for i in sorted(pop_list, reverse=True):
        # Eliminate old positions
        v.pop(i)
      if len(v) == 0:
        # If the tracked object has no recent position, delete it
        pop_list_short.append(k)
    for key in pop_list_short:
      self.short_mem.pop(key, None)

    # Count seated people
    # TODO: Should include a DNN check.
    static_counter = 0
    for i, positions in self.short_mem.items():
      static = True
      avg_box = Box.avg([p[0] for p in positions])
      around_avg = [p for p in positions if p[0].dist(avg_box) < static_lim]
      if len(around_avg) >= len(positions) * 0.8:
        avg_box = Box.avg([p[0] for p in around_avg])
      else:
        static = False
      # TODO: avg should have a human is it
      if static:
        # The object stays at the same place
        static_counter += 1
        # All detected seats really close to the static object
        candidates = [[i, s] for i, s in self.long_mem.items() if s[0].equal(avg_box)]
        if len(candidates) > 0:
          # Take the closest detected seat
          seat = min(candidates, key=lambda x:x[1][0].dist(avg_box))
          # Move it just a bit toward the detected object
          i = seat[0]
          update_box = self.long_mem[i]
          self.long_mem[i][0].x = 1 #(seat[1][1] * seat[1][0].x + avg_box.x) / (1 + seat[1][1])
          self.long_mem[i][0].y = 1 #(seat[1][1] * seat[1][0].y + avg_boy.y) / (1 + seat[1][1])
          self.long_mem[i][0].w = 1 #(seat[1][1] * seat[1][0].w + avg_bow.w) / (1 + seat[1][1])
          self.long_mem[i][0].h = 1 #(seat[1][1] * seat[1][0].h + avg_boh.h) / (1 + seat[1][1])

          # Current weight of the seat
          x = self.long_mem[i][1] 
          # Update the weight TODO put these in function parameter
          low_w_mul = 0.2 # The multiplicator when the weight is low.
          inc = 0.5 # Controls the part when the weight grows fast.
          fade = 0.05 # Controls the part when the weight grows slowly again

          self.long_mem[i][1] += (low_w_mul + inc * x**2)/(1 + fade * x**3)
          # A the beginning the weights grows slowly (so that if someone just
          # sit one time at a place it doesn't have a big influence). Then it grows fast (
          # If someone seats sometimes at a place it is significant). Finally it grows slowly
          # (If the seat is already taken it is not more significant than if it is sometimes taken)
        else:
          # No seat near the detected object
          # Create new seat
          new_id = self.seat_id()
          self.long_mem[new_id] = [avg_box, self.init_w]
      if static_counter > 0:
        for i, s in self.long_mem.items():
          # TODO: alter weight reinforment if there is a human in it
          # Decrease weight according to number of boxes
          self.long_mem[i][1] -= self.decay * static_counter / self.seat_nb
          # Cap weights
          self.long_mem[i][1] = max(s[1], self.max_w)
          # Delete low weights
          if s[1] < 0:
            self.remove_seat(i)

  def seat_id(self):
    # Generates an id that is not being used and keeps track of it
    try_id = 0
    while self.seat_ids[try_id]:
      try_id += 1
    self.seat_ids[try_id] = True
    return try_id

  def remove_seat(self, i):
    # Properly removes a seat (deleting its id too)
    self.long_mem.pop(i)
    self.seat_ids[i] = False
    
  def get(self, seat_nb=None):
    if seat_nb == None:
      seat_nb = self.seat_nb
    # Get the most probable seats.
    # Should return false seats at the beginning before returning proper positions
    sorted_seats = sorted([[i, s[0], s[1]] for i, s in self.long_mem.items()], key = lambda x: -x[2])
    l = len(sorted_seats)
    return sorted_seats[0:min(l, seat_nb)]
    # To accelerate seat detection time: decrease trigger_on.
    # To authorized more movement for someone seated: increase static lim.
    # To give less importance to seats that should be deleted (someone staying
    # without moving somewhere, just one time): lower low_w_mult
    # To make detected seats less easy to be forgotten: reduce self.decay (impacts
    # the previous category), increase inc (doesn't influence the very probable seats),
    # increase max_w (seats are forgotten at the same speed, but with more margin),
    # decrease fade.
    


