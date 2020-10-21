import argparse
import imutils
import cv2


image = cv2.imread("friends.png")
(h,w,d) = image.shape

#Rotating an image
center = (w//2,h//2)
M=cv2.getRotationMatrix2D(center, -45, 1.0)
rotated = cv2.warpAffine(image,M,(w,h))
cv2.imshow("OpenCV Rotation",rotated)
cv2.waitKey(0)

rotated = imutils.rotate_bound(image,45)
cv2.imshow("Imutils Bound Rotation", rotated)
cv2.waitKey(0)

#Smoothing an image
blurred = cv2.GaussianBlur(image,(11,11),0)
cv2.imshow("Blurred",blurred)
cv2.waitKey(0)

#Counting objects
ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, 
   help="path to input image")
args = vars(ap.parse_args())

#Converting an image to grayscale
image = cv2.imread(args["image"])
cv2.imshow("Image", image)
cv2.waitKey(0)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray",gray)
cv2.waitKey(0)

#Edge detection
edged = cv2.Canny(gray,30 ,150)
cv2.imshow("Edged", edged)
cv2.waitKey(0)

#Thresholding
thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]

#Detecting and Drawing Contours
cnts = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
output = image.copy()

for c in cnts:
    cv2.drawContours(output, [c], -1, (240,0,159),3)
    cv2.imshow("Contours", output)
    cv2.waitKey(0)

text = "I found {} objects!".format(len(cnts))
cv2.putText(output, text, (10,25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(240,0,159),2)
cv2.imshow("Contours",output)
cv2.waitKey(0)