from keras.preprocessing import image
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers import Conv2D, MaxPooling2D
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical
from keras.utils.vis_utils import plot_model
import keras
import numpy as np
import matplotlib.pyplot as plt
#モデルの保存
import os
#混同行列表示
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

'''
BTSのメンバーを分類するモデルを作成
'''

MEMBER = ["J-HOPE","JIMIN","JIN","JUNGKOOK","RM","SUGA","V"]
#画像とラベルを保存
X = []
y = []
#メンバーの画像を読み込み
def read_dir_img(member,label):
    for i in os.listdir("member_face_triming_0323/" + member + "/"):
        X.append(image.img_to_array(image.load_img("member_face_triming_0323/" + member+"/"+i,target_size=(150,150,1),grayscale=True)))
        y.append(label)

for j in range(len(MEMBER)):
    read_dir_img(MEMBER[j],j)

# データのロード
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
#list型をnumpy配列に変換
X_train = np.array(X_train)
X_test = np.array(X_test)
#正解ラベルをOnne-hot形式に変換
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# モデルの定義
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(3, 3),input_shape=(150,150,1)))
model.add(Activation('relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(32))
model.add(Dropout(0.25))
model.add(Dense(7,activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
          batch_size=128,
          epochs=150,
          verbose=1,
          validation_data=(X_test, y_test))

# 精度の評価
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# 予測（テストデータの先頭の10枚）
pred = np.argmax(model.predict(X_test[0:10]), axis=1)
y_pred = model.predict_classes(X_test)

# データの可視化（テストデータの先頭の10枚）
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(X_test[i].reshape((150,150)), 'gray')
    plt.title(MEMBER[pred[i]],fontsize=10)
plt.suptitle("10 images of test data",fontsize=20)
plt.show()

#混同行列表示
y_true = np.argmax(y_test,axis=1)
cm = confusion_matrix(y_true,y_pred)
cm = pd.DataFrame(data=cm,index=MEMBER,columns=MEMBER)
sns.heatmap(cm,square=True,cbar=True,annot=True,cmap='Blues')
plt.show()

##accとval_accのプロット
plt.plot(history.history["accuracy"], label="acc", ls="-", marker="o")
plt.plot(history.history["val_accuracy"], label="val_acc", ls="-", marker="x")
plt.ylabel("accuracy")
plt.xlabel("epoch")
plt.legend(loc="best")
plt.show()

model.summary()

#resultsディレクトリを作成
result_dir = 'results'
if not os.path.exists(result_dir):
    os.mkdir(result_dir)
# 重みを保存
model.save(os.path.join(result_dir, 'bts_model.h5'))