import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 左右反転の水増しのみ使用
def scratch_image(img, flip=True, thr=True, filt=True, resize=False, erode=False):
    # 水増しの手法を配列にまとめる
    methods = [flip, thr, filt, resize, erode]

    # flip は画像の左右反転
    # thr  は閾値処理
    # filt はぼかし
    # resizeはモザイク
    # erode は縮小
    #     をするorしないを指定している
    # 
    # imgの型はOpenCVのcv2.read()によって読み込まれた画像データの型
    # 
    # 水増しした画像データを配列にまとめて返す

    # 画像のサイズを習得、ぼかしに使うフィルターの作成
    img_size = img.shape
    filter1 = np.ones((3, 3))

    # オリジナルの画像データを配列に格納
    images = [img]

    # 手法に用いる関数
    scratch = np.array([
        #画像の左右反転のlambda関数を書いてください
        lambda x: cv2.flip(x, 1),
        #閾値処理のlambda関数を書いてください
        lambda x: cv2.threshold(x, 150, 255, cv2.THRESH_TOZERO)[1],  
        #ぼかしのlambda関数を書いてください
        lambda x: cv2.GaussianBlur(x, (5, 5), 0),
        #モザイク処理のlambda関数を書いてください
        lambda x: cv2.resize(cv2.resize(x,(img_size[1]//5, img_size[0]//5)), (img_size[1], img_size[0])),
        #縮小するlambda関数を書いてください
        lambda x: cv2.erode(x, filter1)  
    ])

    # 関数と画像を引数に、加工した画像を元と合わせて水増しする関数
    doubling_images = lambda f, imag: (imag + [f(i) for i in imag])

    # doubling_imagesを用いてmethodsがTrueの関数で水増ししてください

    for func in scratch[methods]:
        images = doubling_images(func,  images)

    return images


for fold_path in glob.glob('C:/Users/kato/Desktop/from_app_ver2/FaceEdited/*/*'):
    imgs = glob.glob(fold_path + '/*.jpg')
    print(imgs)
    # 顔切り取り後の、画像保存先のフォルダ名
    save_path = fold_path.replace('FaceEdited','Aug')
    
    # 保存先のフォルダがなかったら、フォルダ作成
    if not os.path.exists(save_path):
        print(save_path)
        os.makedirs(save_path)

    # 画像ごとに処理
    for i, img_path in enumerate(imgs,1):

        # 画像ファイル名を取得
        base_name = os.path.basename(img_path)
        print(base_name)
        # 画像ファイル名nameと拡張子extを取得
        name,ext = os.path.splitext(base_name)
        print(name + ext)

        # 画像ファイルを読み込む
        img = cv2.imread(img_path, 1)
        scratch_images = scratch_image(img)



        for j, im in enumerate(scratch_images):
             #認識結果の保存
            file_name = "/{:0=2}_{:0=2}.jpg".format(i,j)
            print(file_name)
            cv2.imwrite(save_path+file_name, im)