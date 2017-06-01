import cv2
import time

def blur(img, fgbg, blur_kernel, denoise_kernel):
  # Sequence of processing to remove useless information on the picture
  # Removes colors
  t = time.time()
  grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  # Removes background, put forground in white
  fgmask = fgbg.apply(grey)
  # Denoises the fgbg picture
  # Blurr the found foreground to connect close patches corresponding to the same object
  blurred = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, blur_kernel)
  # Removes small independent moving objects.
  denoised = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, denoise_kernel)
  # Blurr again
  #blurred = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, blur_kernel)
  # Smooth out the image, without altering edges.
  #cleaned = cv2.bilateralFilter(blurred, 10, 75, 75)
  return denoised

def bound(img):
  # delimit patches in the image, bound them by rectangles
  contours = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2]
  rects = [cv2.boundingRect(cnt) for cnt in contours]
  return rects
