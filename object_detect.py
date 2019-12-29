# -*- coding: utf-8 -*-
"""
Created on Mon May 27 21:00:21 2019

@author: rbdud
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

fileName = '1st_school_tour_playground.avi'
cap = cv2.VideoCapture(fileName)
winname = "result"
winname2 = "CLick point"
cv2.namedWindow(winname)
cv2.namedWindow(winname2)
# 마우스로 클릭한 좌표가 담기는 arr
arr = []
# 가우시안 블러사이즈
blurrSize = 3
# 매치카운트
MIN_MATCH_COUNT = 10
# 마우스 클릭이 4번 되면 매칭이 시작된도록 플래그
is_start = False
# 마우스 포인터로 크롭된 이미지를 담을 변수
target = None
# 마우스 콜백
# number of frames needed to update
UR = 3
# ratio of mean
# B*mean +(1-B)*lastmean
B = 0.8
time = 10
distanceThresh = 250


counter = 0
maxCoordinate = 0
minCoordinate = 1000000
meanP = (0, 0)
lastMeanP = (0, 0)
meanX = 0
meanY = 0
width = height = 0


def draw(event, x, y, flags, param):
    global is_start, target, width, height, lastMeanP
    # 왼쪽 마우스 버턴이 눌러졌을 때
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        arr.append([x, y])
        # 총 2번 클릭
        if (len(arr) == 2):
            left = right = top = bottom = None
            if arr[0][0] < arr[1][0]:
                left = arr[0][0]
                right = arr[1][0]
            else:
                right = arr[0][0]
                left = arr[1][0]
            if arr[0][1] < arr[1][1]:
                top = arr[0][1]
                bottom = arr[1][1]
            else:
                top = arr[0][1]
                bottom = arr[1][1]
            width = right - left
            height = bottom - top
            print(height, width)
            # 이미지 크롭
            target = img1[top:bottom, left:right]
            lastMeanP = (left + width / 2 ,top + height/2)
            # 크롭된 이미지가 저장된다.
            arr.clear()
            is_start = True

cv2.setMouseCallback(winname2, draw)
sift = cv2.xfeatures2d.SIFT_create()
# get image
ret, img1 = cap.read()
cv2.imshow(winname2, img1)
cv2.waitKey(0)
cv2.destroyWindow(winname2)

updateCount = 0
goodHistory = []

while True:
    ret, img1 = cap.read()
    if ret == False:
        break

    if is_start:
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(target, None)
        matcher = cv2.DescriptorMatcher_create(
            cv2.DescriptorMatcher_FLANNBASED)
        matches = matcher.knnMatch(des1, des2, k=2)

        good = []
        good_for_homography = []
        for m, n in matches:
            if m.distance < 0.3*n.distance:
                good.append([m])
                good_for_homography.append(m)
        if len(good) > MIN_MATCH_COUNT:
            # 수정
            src_pts = np.float32(
                [kp1[m.queryIdx].pt for m in good_for_homography])
            # 수정
            dst_pts = np.float32(
                [kp2[m.trainIdx].pt for m in good_for_homography])
            # 수정

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()

            h, w, c = img1.shape
            pts = np.float32(
                [[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            target = cv2.polylines(
                target, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            # print "Not enough matches are found - %d/%d" % (len(good),MIN_MATCH_COUNT)
            matchesMask = None

        # print(target)
        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=matchesMask,  # draw only inliers
                           flags=2)

        img3 = cv2.drawMatches(
            img1, kp1, target, kp2, good_for_homography, None, flags=2)
        # 원그리기
        cv2.circle(img3, meanP, 60, (0, 255, 0), 3)
        cv2.imshow(winname, img3)
        counter += 1
        if counter == UR:
            meanX = 0
            meanY = 0
            minusCount = 0
            for i in src_pts:
                if np.abs(lastMeanP[0] - i[0]) < distanceThresh and np.abs(lastMeanP[1] - i[1]) < distanceThresh:
                    meanX += i[0]
                    meanY += i[1]
                else:
                    minusCount += 1
            if minusCount-len(src_pts) != 0:
                meanX = int(int(meanX)/(len(src_pts)-minusCount))
                meanY = int(int(meanY) / (len(src_pts) - minusCount))
                meanP = (meanX, meanY)
                meanX = int(meanX * B + meanX * (1 - B))
                meanY = int(meanY * B + meanY * (1 - B))
                target = img1[int(meanY - height/2): int(meanY + height/2),
                            int(meanX - width/2): int(meanX + width/2)]
                counter = 0
                lastMeanP = meanP
    else:
        cv2.imshow(winname, img1)

    # esc입력시 종료
    if cv2.waitKey(time) & 0xFF == 27:
        break

if __name__ == '__main__':
    cap.release()
    cv2.destroyAllWindows()