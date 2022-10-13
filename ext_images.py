import cv2
import glob #ファイル読み込みで使用
import os #フォルダ作成で使用


cascade_path= os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_alt.xml"
)
face_cascade = cv2.CascadeClassifier(cascade_path)

for fold_path in glob.glob('./Original/*'):
    imgs = glob.glob(fold_path + '/*.jpg')
    # 顔切り取り後の、画像保存先のフォルダ名
    save_path = fold_path.replace('Original','Face')
    
    # 保存先のフォルダがなかったら、フォルダ作成
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    # 顔だけ切り取り→保存
    # 画像ごとに処理
    for i, img_path in enumerate(imgs,1):
        print(img_path)
        img = cv2.imread(img_path)


        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=2, minSize=(30, 30))

        #print(facerect)
        color = (255, 255, 255) #白
        if len(facerect) > 0:

            #検出した顔を囲む矩形の作成
            for rect in facerect:
                print(rect)
                cv2.rectangle(img, tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]), color, thickness=2)
                print(tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]))
            #認識結果の保存
            file_name = "/{}.jpg".format(i)
            print(file_name)
            cv2.imwrite(save_path+file_name, img)
        else :
            print("{}.jpg_skip".format(i))
