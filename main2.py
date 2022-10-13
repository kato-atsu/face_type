import os
import cv2
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing import image

import numpy as np


classes = ["塩顔","醤油顔", "ソース顔"]
num_classes = len(classes)
image_size = 64

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# 顔を検出して顔部分の画像（64x64）を返す関数
def detect_face(img):
    # 画像をグレースケールへ変換
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # カスケードファイルのパス
    cascade_path= os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_alt.xml"
)
    # カスケード分類器の特徴量取得
    cascade = cv2.CascadeClassifier(cascade_path)
    # 顔認識
    faces=cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=1, minSize=(10,10))

    # 顔認識出来なかった場合
    if len(faces) == 0:
        face = faces
    # 顔認識出来た場合
    else:
        # 顔部分画像を取得
        for x,y,w,h in faces:
            face = img[y:y+h, x:x+w]
        # リサイズ
        face = cv2.resize(face, (image_size, image_size))
    return face


# 学習済みモデルをロードする
model = load_model('./model.h5')




@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('ファイルがありません')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('ファイルがありません')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
            filepath = os.path.join(UPLOAD_FOLDER, filename)

            #受け取った画像を読み込み
            img = cv2.imread(filepath, 1)
            # 顔検出して大きさ64x64
            img = detect_face(img)

            # 顔認識出来なかった場合
            if len(img) == 0:
                pred_answer = "顔を検出できませんでした。他の画像を送信してください。"
                return render_template("index.html",answer=pred_answer)
            # 顔認識出来た場合
            else:
                # 画像の保存
                image_path = UPLOAD_FOLDER + "/face_" + file.filename
                cv2.imwrite(image_path, img)

                img = image.img_to_array(img)

                data = np.array([img])
                #print(data.shape)

                result = model.predict(data)[0]
                print(result)
                r_sorted = np.argsort(result)
                predicted = result.argmax()
                pred_answer = "あなたは"+ str(round(result[r_sorted[-1]]*100,2)) + "%の確率で"+classes[r_sorted[-1]] + " ですね！！！\n"\
                              "もしくは"+ str(round(result[r_sorted[-2]]*100,2)) + "%の確率で"+classes[r_sorted[-2]] + " ですね！！！\n"
                              "もしくは"+ str(round(result[r_sorted[-3]]*100,2)) + "%の確率で"+classes[r_sorted[-3]] + " ですね！！！\n"


                message_comment = "顔を検出できていない場合は他の画像を送信してください"

                return render_template("index.html",answer=pred_answer, img_path=image_path, message=message_comment)


    return render_template("index.html",answer="")


if __name__ == "__main__":
    app.run()