import cv2
import glob #ファイル読み込みで使用
import os #フォルダ作成で使用


cascade_path= os.path.join(
    cv2.data.haarcascades, "haarcascade_frontalface_alt.xml"
)
face_cascade = cv2.CascadeClassifier(cascade_path)



for fold_path in glob.glob('C:/Users/kato/Desktop/from_app_ver2/Original/*/*'):
    imgs = glob.glob(fold_path + '/*.jpg')
    # 顔切り取り後の、画像保存先のフォルダ名
    save_path = fold_path.replace('Original','FaceEdited')
    
    # 保存先のフォルダがなかったら、フォルダ作成
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 顔だけ切り取り→保存
    # 画像ごとに処理
    for i, img_path in enumerate(imgs,1):
    
        img = cv2.imread(img_path)


        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print('{}:read_done'.format(i))
        #カスケード分類器の特徴量を取得する
        cascade = cv2.CascadeClassifier(cascade_path)

        facerect = cascade.detectMultiScale(img_gray, scaleFactor=1.09, minNeighbors=2, minSize=(50, 50))

        #print(facerect)
        color = (255, 255, 255) #白
        print("{}_facerect:".format(i),facerect)
        # 検出した場合
        if len(facerect) > 0 :
            

            #検出した顔を囲む矩形の作成
            for rect in facerect:
                y=rect[0]
                x=rect[1]
                h=rect[2]
                print('rect:',rect)
                print('plot:',tuple(rect[0:2]),tuple(rect[0:2]+rect[2:4]))
                trim = img[y:y+h,x:x+h]
                print(trim.shape[0],trim.shape[1])
                print(trim.shape[1]!=0 )
                print(trim.shape[0]!= 0 and trim.shape[1]!= 0)
                if trim.shape[0]!= 0 and trim.shape[1]!= 0 :
                
                    
                    #認識結果の保存
                    file_name = "/{}.jpg".format(i)
                    print(file_name)
                    cv2.imwrite(save_path+file_name, trim)
                else:
                    print(trim.shape[0],trim.shape[1])
                   
                    
        else:
            
            print("{}:skip".format(i))
            
