import cv2
import numpy as np
import os

MEMBER = ["J-HOPE","JIMIN","JIN","JUNGKOOK","RM","SUGA","V"]
#画像の反転
member = "V"
for i in range(1,217):
    img = cv2.imread("./member_gray_scraping_face_0322/"+ member +"/" + member + "_ (" +  str(i+1) + ")"'.jpg')
    img_flip = cv2.flip(img, 1)
    cv2.imwrite('./member_gray_scraping_face_0322/'+ member +"/"+ member + "_" +  str(i) + '.jpg', img_flip)

