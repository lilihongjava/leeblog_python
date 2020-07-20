# encoding: utf-8
"""
@author: lee
@time: 2020/7/16 11:36
@file: sklearn_lda.py
@desc: 
"""
import os

import joblib

import jieba
import pandas as pd
import re
import numpy as np

# 创建停用词列表
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from lda.util.text_util import stopwords_list, seg_depart


def list_seg_depart(list_seg):
    sw = stopwords_list()
    return [seg_depart(x, sw) for x in list_seg]


# 词频向量
def count_frequency(corpus, in_file):
    if os.path.exists(in_file):
        cnt_vector = joblib.load(in_file)
        cnt_tf = cnt_vector.transform(corpus)
    else:
        cnt_vector = CountVectorizer()
        cnt_tf = cnt_vector.fit_transform(corpus)
        print('主题词袋：', len(cnt_vector.get_feature_names()))
        joblib.dump(cnt_vector, in_file)
    return cnt_tf


# lda模型
def lda_fit_transform(in_model, model_in_data):
    if os.path.exists(in_model):
        lda = joblib.load(in_model)
        res = lda.transform(model_in_data)
    else:
        # LDA主题模型
        lda = LatentDirichletAllocation(n_components=2,  # 主题个数
                                        # max_iter=5,    # EM算法的最大迭代次数
                                        # learning_method='online',
                                        learning_offset=50.,  # 仅仅在算法使用online时有意义，取值要大于1。用来减小前面训练样本批次对最终模型的影响
                                        random_state=0)
        res = lda.fit_transform(model_in_data)
        joblib.dump(lda, in_model)
    return res


if __name__ == '__main__':
    df = pd.read_csv('./data/cnews.csv')
    data = df["content"]
    data_list = data.values.tolist()
    data_list = list_seg_depart(data_list)
    print(data_list)
    cv_file = "./CountVectorizer.pkl"
    cnt_data_list = count_frequency(data_list, cv_file)
    model_file = "./lda_model.pk"
    docres = lda_fit_transform(model_file, cnt_data_list)
    # 文档所属每个类别的概率
    LDA_corpus = np.array(docres)
    print('类别所属概率:\n', LDA_corpus)

    # 预测
    pre_list = ["北美票房综述：《速度与激情4》王牌归位(图)本周综述2009年的第一个“首映性话题”正式诞生！回归系列原点的《速度与激情4》（Fast and Furious），以约7251万美元的夸张成绩高调亮相（该数据打破多项纪录，后文详述），瞬间把影市煮沸！！！Top 12的总和继续攀升近10%，大盘累计约1.52亿＄。这不仅是今年春季档的最火爆周末、自今年冬季情人节档期（总统纪念日周末）以来的最卖座周末，更刷新了影史同期的最好成绩，甚至成为整个4月份的历代最畅销周末（此前的纪录保持者是2005年4月第一周末《地狱男爵1》坐庄时的1.14亿＄）。“2010奥斯卡战线”继续前进。伍迪·艾伦的新片《管用就好》（Whatever Works）和佩德罗·阿莫多瓦的新片《破碎的拥抱》（Broken Embraces）内部试映均获一致好评。另外还有“《口是心非》（Duplicity，托尼·吉尔洛伊执导）的朱莉亚·罗伯茨有望入围金球奖”、“拉斯·冯·特里尔的新片《反基督》（Antichrist）和昆汀·塔伦蒂诺的新片《无良杂牌军》（Inglourious Basterds）均会参赛戛纳电影节”等重磅传闻。同时，关于第81届奥斯卡颁奖典礼的最后一则内幕也在这星期大白于天下。值得一提的还有，“2009奥斯卡终极大盘点”本周完结，最后为您细数本届颁奖典礼的六大看点，以此收尾～ 目录：一、北美票房榜二、新片《速度与激情4》、《冒险乐园》、《大而无当》、《放牛班欢乐颂》、《异形入侵》、《猎杀大行动》、《倒霉蛋也有春天》三、2010奥斯卡战线（二）本周流言[09年总第15期]四、2009奥斯卡终极大盘点（六）五、预告片推荐＋下周预告状元郎《速度与激情4》（Fast and Furious）一举推倒上星期领衔的CG动画《怪兽大战外星人》（Monsters Vs. Aliens），在远远不如对手强势的3461家院线内，以单馆平均近21000＄的超高亩产，收割到约7251万＄的巨大丰收。各位读者朋友们想必也看出来了——对比8500万＄的预算，如此秀逸的数字，即便放在暑期档依旧惊人。《速度与激情》系列的前两部平均1.3亿＄，在第一部结束后文·迪塞尔走人、而第二部独挑大梁的保罗·沃克也在第三部开机前逃之夭夭，于是第三部最终仅有6000万＄出头，还不如最新一集首映三天捞得多。言归正传，回头看《速与激4》的7251万＄首映值，该成绩是：《速度与激情》系列的历代最高首映（远超二代的5047万＄）；“赛车电影”的影史历代最高首映（远超《赛车总动员》的6012万＄）；本年度迄今的最高首映（远超《怪兽大战外星人》的5932万＄和《守望者》的5521万＄）；4月份的影史历代最高首映（远超《愤怒管理》的4222万＄和《恐怖电影4》的4022万＄）；春季档的影史历代最高首映（超越《斯巴达300勇士》的7089万＄和《冰河世纪2》的6803万＄）。……不仅如此，《速度与激情》甚至能在“PG-13作品”内排进影视历代第19位，与《辛普森一家》、《王牌大贱谍3》、《指环王3：王者归来》、《侏罗纪公园2：失落的世界》等猛片称兄道弟。更令人咋舌的是，该片仅在首映当日便进账3011万＄，虽然只在影史首映成绩中排名第21位，却改写了整个春季档（3月／4月）的历史，是“春季首发作品”的影史历代最火爆开画日，同样把《斯巴达300勇士》等一干手下败将抛在身后。尽管周六较之周五跳水约19%，显示不出强韧的后劲——但哪怕首映数字占最终收益的50%，《速度与激情4》仍能创造系列的最高战绩。本来，受第三集票房大惨败的影响，环球对本作已无信心，面对5月底的《飞屋历险记》（Up）和6月底的《变形金刚2：卷土重来》（Transformers: Revenge of the Fallen）的前后夹击，决意退避三舍。因此，早早安排《速度与激情4》从暑期档的6月上旬一路逃奔至此。熟料映期提前后柳暗花明、喜从天降，一举书写了本年度迄今为止的最华丽首映。看来文·迪塞尔、保罗·沃克、米切尔·罗德里格兹、乔丹娜·布雷斯特“四大元老”同时回归的魅力果然不同凡响。热情不减当年的初代粉丝蜂拥而来不说；对《极速赛车手》的卡通路线大失所望的现实主义影迷们，也在第二款预告片的感召下摩肩接踵地迈进映厅。30岁以下的男性观众几乎占据七成以上，难怪《灾难先知》、《12回合》、《飓风营救》、《守望者》等同样主攻相应客源的动作片纷纷大幅跳水。2005年的《超级奶爸》大卖后惨遭《判我有罪》和《巴比伦纪元》两连败的文·迪塞尔、2006年的《南极大冒险》热映后不幸摊上《夺命枪火》和《父辈的旗帜》两度票房砸锅的保罗·沃克、《速度与激情1》结束后只能在TV不朽剧《迷失》中偶尔客串的米切尔·罗德里格兹、2006年暑期搞砸系列第三部而被踢回独立制片界的导演林诣彬……。综观诸位落魄的剧组核心阵容，可知《速度与激情4》本星期的爆棚，着实救活了一大批人，在此谨向各位道一声——恭喜发财！！！按系列的一般规律，下星期跌破55%几乎是板上钉钉的劫数。可尽管如此，本片仍有一定几率连庄——只要《汉娜·蒙塔纳》（Hannah Montana The Movie）那个黄花闺女不“爆发”——实现《守望者》和《怪兽大战外星人》未竟的愿望。保守估计，《速度与激情4》至少有望进账1.5亿＄，至于更高的数字则暂不宜过分乐观。再来看滑落榜眼的《怪兽大战外星人》。不得了，本片的制作成本竟高达1.75亿＄，真是一条可怕的新闻！！！上周末在IMAX巨幕影厅疯狂席卷2500万＄（首映总计5932万＄）的头条记忆犹新，转眼便被天价预算这一瓢冷水浇飞。尽管该片本周末已顺利突破1亿＄关卡，但无论《冰河世纪2》还是《马达加斯加2》的同期数据，都已把《怪兽大战外星人》远远甩在身后。再结合《冰2》最终1.95亿＄、《马2》最终1.8亿＄的实际情况，……也就是说，《怪战外》冲击2亿＄的梦想果然如我上周所言，破灭了！？虽然这星期受了新人王《速度与激情4》不少气，但我坚信，凭借全家共赏的绵长内力，《怪兽大战外星人》的总销量很可能反超新科状元——至于最终的着陆点，还是保持上周的看法，八成会在1.5亿＄至1.8亿＄之间吧。梦工厂的各位粉丝们，千万别忘了到我国院线去支持这部CG卡通片，尽管3D眼镜有点烦人，但作品本身绝对十分娱乐！上周探花《康涅狄格鬼屋事件》的“出口调查”反映出一个重大现象。虽然是恐怖片，却有近62%的观众是17岁至24岁的年轻女性。原来去年末至今的恐怖热浪，原动力竟是女性粉丝。究竟是《暮光之城》培养的结果，还是TV小腐剧《神秘力量》的两大男主角先后出入《血腥情人节》和《黑色星期五》刺激而成？……总之，此话题值得玩味。Top 12的其他各位也没什么好说，大家不妨自己看数据。惟有本周亮相的新片《冒险乐园》（Adventureland）令人心疼，枉有超越《太坏了》的高媒体评价和来自《暮光之城》的人气女主角，依然在讨喜的《怪兽大战外星人》、《寻找伴郎》、《地球奶爸》等强敌的围追堵截下运营惨淡。首映周末仅在1862家电影院攫金601万＄的战果已足够丧气，更悲情的是，与反响奇佳的现实相对应的居然是“周五＞周六＞周日”的短命销售曲线，但愿下星期的走势能有所好转，不然还真挺奇怪的。"]
    pre_list = list_seg_depart(pre_list)
    pre_cnt_data_list = count_frequency(pre_list, cv_file)
    pre_docres = lda_fit_transform(model_file, pre_cnt_data_list)
    print('预测数据概率:\n', np.array(pre_docres))
