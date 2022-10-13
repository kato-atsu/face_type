import os
import glob
from tkinter import image_types
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Dropout, Flatten, Input
from keras.applications.vgg16 import VGG16
from keras.models import Model, Sequential
from keras import optimizers


# 各クラス配列格納
type_list = ["shoyu", "sio", "sosu"]


# 各クラスの画像ファイルパスを配列で取得する関数
def get_path_type(type):
  path_type = glob.glob('./Aug/' + type + '/*/*')
  return path_type

#リサイズ時のサイス指定
img_size = 64


# 各クラスの画像データndarray配列を取得する関数
def get_img_type(type):
  path_type = get_path_type(type)

  img_type = []
  for i in range(len(path_type)):
    # 画像の読み取り、64にリサイズ
    img = cv2.imread(path_type[i])
    img = cv2.resize(img, (img_size, img_size))
    # img_typeに画像データのndarray配列を追加していく
    img_type.append(img)
  return img_type 



# 各クラスの画像データを合わせる
X = []
y = []
for i in range(len(type_list)):
    print(type_list[i] + ":" + str(len(get_img_type(type_list[i]))))
    X += get_img_type(type_list[i])
    y += [i]*len(get_img_type(type_list[i]))
X = np.array(X)
y = np.array(y)

print(X.shape)

# ランダムに並び替え
rand_index = np.random.permutation(np.arange(len(X)))

# 上記のランダムな順番に並び替え
X = X[rand_index]
y = y[rand_index]

# データの分割（トレインデータが8割）
X_train = X[:int(len(X)*0.8)]
y_train = y[:int(len(y)*0.8)]
X_test = X[int(len(X)*0.8):]
y_test = y[int(len(y)*0.8):]

# one-hotベクトルに変換
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# モデル
input_tensor = Input(shape=(64, 64, 3))
vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=vgg16.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(len(type_list), activation='softmax'))

model = Model(inputs=vgg16.input, outputs=top_model(vgg16.output))

# vgg16の重みの固定
for layer in model.layers[:15]:
    layer.trainable = False

# モデルの読み込み
model.compile(loss='categorical_crossentropy',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

model.summary()
history = model.fit(X_train, y_train, batch_size=64, epochs=50,  validation_data=(X_test, y_test))

# モデルの保存
model.save('model.h5')


# 精度の評価（適切なモデル名に変えて、コメントアウトを外してください）
scores = model.evaluate(X_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

# acc, val_accのプロット
plt.plot(history.history['acc'], label='acc', ls='-')
plt.plot(history.history['val_acc'], label='val_acc', ls='-')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(loc='best')
plt.savefig('figure01.jpg')