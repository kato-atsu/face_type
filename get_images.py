import os
from icrawler.builtin import	BingImageCrawler


sio = ["坂口健太郎","中村倫也","杉野遥亮","神木隆之介","西島秀俊","高橋一生",
"綾野剛","向井理","渡辺翔太","窪田正孝","神木隆之介","福士蒼汰","星野源","長谷川博己"
,"伊藤健太郎","松田翔太","及川光博"]


shoyu = ["小栗旬","向井理","松坂桃李","佐藤健","窪田正孝","平野紫耀",
"山下智久","東山紀之","福山雅治","鈴木伸之","吉沢亮","綾野剛","水嶋ヒロ","岡田准一"
,"中川大志","大野智","伊藤健太郎"]

sosu = ["竹野内豊","新田真剣佑","長瀬智也","斎藤工","岡田准一","松本潤","水嶋ヒロ","玉山鉄二","山田孝之","伊藤英明","大谷亮平","阿部寛","桐谷健太"]

class_ =[sio,shoyu,sosu]

def get_im(save_path,name):
	#1---任意のクローラを指定
	crawler = BingImageCrawler(storage={"root_dir":save_path + name})
	#2---検索内容の指定
	crawler.crawl(keyword=name, max_num=30)

for name in class_:
	if name == sio:
		save_path =  "./Original/塩/"
		if not os.path.exists(save_path):
			os.mkdir(save_path)

	elif name == shoyu:
		save_path =  "./Original/醤油/"
		if not os.path.exists(save_path):
			os.mkdir(save_path)
	else:
		save_path =  "./Original/ソース/"
		if not os.path.exists(save_path):
			os.mkdir(save_path)
	for kojin in name:
		print(kojin)
		#1---任意のクローラを指定
		crawler = BingImageCrawler(storage={"root_dir":save_path + kojin})
		#2---検索内容の指定
		crawler.crawl(keyword=kojin, max_num=30)