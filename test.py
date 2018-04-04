import cv2
import numpy as np

img_color = cv2.imread('test9.jpg', cv2.IMREAD_COLOR)

img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY) # img_color 컬러 이미지를 그레이 스케일 img_gray 이미지로 변환

kernel = np.ones((3, 3), np.uint8)
# 이미지 윤곽선 추출
gradient = cv2.morphologyEx(img_gray, cv2.MORPH_GRADIENT, kernel)

# 이미지 이진화(흑백처리)
threshold = cv2.adaptiveThreshold(gradient, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 10)

# 직선찾아 없애기
minLineLength = 100
maxLineGap = 60
lines = cv2.HoughLinesP(threshold, 1, np.pi/180, 100, minLineLength, maxLineGap)

for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(threshold, (x1, y1), (x2, y2), (0, 0, 0), 2) #검정색을 칠함

#글자 윤곽찾기
kernel2 = np.ones((9, 9), np.uint8)
img_close = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel2)

#글자부분 찾기
img_con, contours, hierachy = cv2.findContours(img_close, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# img_contours = cv2.drawContours(img_color, contours, -1, (0, 255, 0), 2)

# Straight Rectangle
for i in range(len(contours)):
    cnt = contours[i]
    x, y, w, h = cv2.boundingRect(cnt)
    img1 = cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)# green



#cv2.namedWindow('gray image', cv2.WINDOW_NORMAL)

#cv2.imshow('gray image', img_gray)
#cv2.imshow('gray image', gradient)
#cv2.imshow('gray image', threshold)

#cv2.imwrite('gray.jpg', img_gray)
#cv2.imwrite('gradient1.jpg', gradient)
#cv2.imwrite('threshold2.jpg', threshold)
#cv2.imwrite('houghlines.jpg', threshold)
#cv2.imwrite('morph_close.jpg', img_close)
cv2.imwrite('contours9.jpg', img1)

cv2.waitKey(0)