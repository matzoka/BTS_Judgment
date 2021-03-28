import cv2
import numpy as np
import os

MEMBER = ["J-HOPE","JIMIN","JIN","JUNGKOOK","RM","SUGA","V"]
#カラー画像を白黒に変換する
member = "V"
for i in range(0,58):
    img = cv2.imread("./member_scraping/"+ member +"/"+ str(i) + '.jpg')
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite('./member_gray_0320/'+ member +"/"+ str(i) + '.jpg', img_gray)

