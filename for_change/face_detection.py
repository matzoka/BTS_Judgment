import cv2
import os
from keras.preprocessing import image

MEMBER = ["J-HOPE","JIMIN","JIN","JUNGKOOK","RM","SUGA","V"]
#メンバーの画像から顔だけをトリミング
def read_img_and_trimming(member):
    #各メンバー296枚
    for i in range(1,411):
        img = cv2.imread("./member_gray_0320/"+ member + "/" + member  + '_ (' + str(i) + ').jpg')
        #print("./member_gray_0322/"+ member + "/" + member  + '_ (' + str(i) + ').jpg')
        image_flip = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cascade_path = 'haarcascade_frontalface_default.xml'
        cascade = cv2.CascadeClassifier(cascade_path)
        face_list = cascade.detectMultiScale(image_flip, minSize=(20, 20))
        for j,(x, y, w, h) in enumerate(face_list):
            trim = image_flip[y: y+h, x:x+w]
            cv2.imwrite('./member_gray_scraping_face_0322/'+ member +'/' + member + '_(' + str(i) +  ').jpg', trim)

read_img_and_trimming("V")
'''
for j in range(len(MEMBER)):
    read_img_and_trimming(MEMBER[j])
'''