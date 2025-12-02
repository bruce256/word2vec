import pandas as pd
from openai import OpenAI
import numpy as np
import time

data = """华为Mate60 Pro 5G手机限时直降500元
小米Redmi Note 13新品上市送耳机
苹果AirPods Pro二代官方正品立省300
联想小新Pad Pro平板电脑学生优惠价
三星Galaxy S24 Ultra旗舰机赠原装壳
OPPO Reno11 5G手机下单即送快充套装
vivo X100 Pro影像旗舰限时特价抢购
索尼WH-1000XM5降噪耳机双12抢先购
戴尔灵越14英寸轻薄本教育折扣特惠
佳能EOS R50微单相机套装含镜头
任天堂Switch OLED游戏机现货秒发
一加Ace 3V性能神机直降200元
华硕天选5 Pro游戏本学生专享价
罗技G502 HERO电竞鼠标限时五折
飞利浦电动剃须刀3系数码屏显款
Anker 100W氮化镓快充多口充电器
大疆Mini 4 Pro无人机航拍套装促销
机械革命蛟龙16Pro高性能游戏本
西部数据1TB移动固态硬盘特价包邮
雷蛇黑寡妇V4机械键盘RGB灯效版
惠普星14青春版轻薄笔记本电脑
绿联Type-C扩展坞多功能HUB
小米手环8 NFC版健康监测智能表
三星T7 Shield 2TB移动固态硬盘
红魔9 Pro游戏手机散热背夹套装
倍思65W快充充电宝自带线便携款
华为FreeBuds 5i无线蓝牙耳机降噪
联想拯救者Y7000P 2024电竞本
闪迪128GB高速MicroSD存储卡十片装
网易有道词典笔X5学生学习神器
罗技MX Keys无线办公键盘静音设计
金士顿DDR4 16GB台式机内存条
九号电动滑板车F系列折叠便携款
海信E8K 85英寸ULED电视以旧换新
雷柏VT9PRO无线游戏鼠标高性价比
七彩虹RTX 4070 Ti战斧显卡现货
飞傲M11 Plus ESS便携音乐播放器
小米路由器AX9000 WiFi6电竞级
科大讯飞智能录音笔SR502会议神器
爱奇艺奇遇Dream VR一体机家庭影院
三星Odyssey G7曲面电竞显示器
绿巨能笔记本支架铝合金可调节
安克Soundcore Liberty 4 NC耳机
华为Watch GT4智能手表运动健康
影驰GeForce RTX 4060金属大师显卡
小度智能音箱带屏版语音控制
铁三角ATH-M50x专业监听耳机
飞利浦27英寸2K IPS显示器办公神器
罗技C920高清网络摄像头直播推荐
联想ThinkBook 16+ 2024锐龙版轻薄本
"""
data2 = """春夏新款法式碎花连衣裙限时5折包邮
高腰显瘦牛仔裤女夏季薄款直筒裤特价
小众设计感腋下包女2025新款百搭单肩
真丝睡衣女夏季冰丝家居服套装清仓
韩版宽松T恤女短袖纯棉百搭打底衫
大容量托特包通勤女包简约上班手提包
蕾丝内衣女无钢圈聚拢舒适文胸套装
仙女风雪纺衬衫女春夏飘带气质上衣
小香风菱格链条包女轻奢百搭斜挎包
冰丝防晒开衫女夏薄款空调衫外搭
高弹力瑜伽裤女健身紧身裤提臀显瘦
纯棉吊带背心女内搭打底基础款多色
轻便旅行箱20寸登机箱万向轮静音
法式复古半身裙高腰A字中长裙夏
情侣睡衣女纯棉家居服套装可单买
小方包女2025新款链条包百搭斜跨
宽松阔腿裤女垂感高腰显瘦九分裤
莫代尔家居服女短袖套装亲肤透气
防晒渔夫帽女夏遮脸大檐可折叠帽
真皮钱包女短款小清新卡包零钱夹
针织开衫女春秋季薄款外搭短外套
妈妈装连衣裙中年女装气质改良旗袍
通勤西装外套女春夏季薄款小西装
透明果冻包女夏PVC手提斜挎小包包
高腰收腹内裤女无痕纯棉三角裤5条
亚麻连衣裙女夏宽松文艺森系长裙
超轻老花托特包大容量通勤手提包
吊带连衣裙女夏V领显瘦碎花长裙
防走光打底背心女夏季无袖内搭衫
小众设计师耳环女轻奢气质耳饰套装
蕾丝边睡裙女夏季性感真丝吊带裙
韩版帆布鞋女平底百搭低帮休闲鞋
羊皮斜挎包女软皮迷你链条小包包
冰丝凉感打底裤女夏薄款高腰紧身
简约纯色围巾女春秋百搭丝巾配饰
孕妇连衣裙夏装宽松显瘦哺乳裙
日系学院风双肩包女学生书包可爱
无痕文胸女聚拢防下垂美背内衣
雪纺阔腿裤女高腰垂感显腿长夏裤
防晒冰袖女夏季骑行袖套护手臂
轻奢真皮手拿包女晚宴小方包
纯棉长袖睡衣女秋冬季加厚家居服
百搭小白鞋女厚底增高休闲板鞋
高级感金属链条包女小众设计感
蕾丝罩衫女夏透视外搭防晒开衫
大号化妆包防水便携旅行洗漱包
韩版高腰短裤女夏牛仔热裤显腿长
真丝眼罩女睡眠遮光冰感助眠神器
小香风毛呢外套女秋冬短款粗花呢
时尚腰包女潮牌斜跨胸包轻便出行"""

data += data2

product_names = data.strip().split('\n')
df = pd.DataFrame({'product_name': product_names})
print(df.head())


client = OpenAI(
    api_key="sk-**",  # 请替换为你在DashScope获取的真实API密钥
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # Qwen的兼容端点
)
EMBEDDING_MODEL = "text-embedding-v3"

def get_embedding(text, model=EMBEDDING_MODEL):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# search through the reviews for a specific product
def search_product(df, query, n=3, pprint=True):
    start_time = time.time()  # 记录开始时间
    query_embedding = get_embedding(
        query
    )
    df["similarity"] = df.embedding.apply(lambda x: cosine_similarity(x, query_embedding))

    results = (
        df.sort_values("similarity", ascending=False)
        .head(n)
        .product_name
    )
    if pprint:
        for r in results:
            print(r)
    end_time = time.time()  # 记录结束时间
    elapsed_time = end_time - start_time  # 计算耗时
    print(f"搜索耗时: {elapsed_time:.2f} 秒")
    return results

embeddings = []

for product_name in product_names:
    embeddings.append(get_embedding(product_name))

df["embedding"] = embeddings

print("embedding 结束")

results = search_product(df, "小米手机", n=5)
