from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, Activation, Input
from keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential, load_model, Model
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import optimizers
import keras
import numpy as np
import matplotlib.pyplot as plt
#モデルの保存
import os

'''
BTSのメンバーを分類するモデルを作成
'''

MEMBER = ["J-HOPE","JIMIN","JIN","JUNGKOOK","RM","SUGA","V"]
#画像とラベルを保存
X = []
y = []
#メンバーの画像を読み込み
def read_dir_img(member,label):
    for i in os.listdir("./member_copy_and_test/" + member + "/"):
        #print("./member_gray/" + member+"/"+i)
        #print(image.img_to_array(image.load_img("./member_gray/" + member+"/"+i)).shape)
        X.append(image.img_to_array(image.load_img("./member_copy_and_test/" + member+"/"+i,target_size=(150,150,1),grayscale=True)))
        y.append(label)

for j in range(len(MEMBER)):
    read_dir_img(MEMBER[j],j)

# データのロード
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
#list型をnumpy配列に変換
X_train = np.array(X_train)
X_test = np.array(X_test)
#正解ラベルをOnne-hot形式に変換
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

input_tensor = Input(shape=(150, 150, 1))
vgg16 = VGG16(include_top=False, weights=None,input_tensor=input_tensor,input_shape=(100,100,1))

# モデルの定義
model = Sequential()
model.add(Flatten(input_shape=vgg16.output_shape[1:]))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(BatchNormalization())
model.add(Dense(7, activation='softmax'))

model = Model(inputs=vgg16.input,outputs=model(vgg16.output))

for layer in model.layers[:19]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4,momentum=0.9),
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=128,
          epochs=50,
          verbose=1,
          validation_data=(X_test, y_test))

# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# 予測（テストデータの先頭の10枚）
pred = np.argmax(model.predict(X_test[0:10]), axis=1)

# データの可視化（テストデータの先頭の10枚）
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape((100,100)), 'gray')
    plt.title(MEMBER[pred[i]],fontsize=5)
plt.suptitle("10 images of test data",fontsize=20)
plt.show()

model.summary()

#resultsディレクトリを作成
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
# 重みを保存
model.save(os.path.join(result_dir, 'bts_model.h5'))