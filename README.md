# IJCAI-18-
## 背景介绍

   搜索广告是一种常见的互联网营销方式，商家（广告主）根据商品特点自主购买特定的关键词，当用户输入这些关键词时相应的广告商品就会展示在用户看到的页面中。随着互联网的快速发展，搜索广告和电商广告在互联网广告中的占比越来越高，成为互联网行业最主要的商业模式之一。与此同时，搜索广告以其巨大的商业价值和研究价值吸引了大量的专家学者，在学术界得到了广泛的研究。\
   搜索广告的转化率，作为衡量广告转化效果的指标，从广告创意、商品品质、商店质量等多个角度综合刻画用户对广告商品的购买意向，即广告商品被用户点击后产生购买行为的概率。举例来说，用户在淘宝搜索栏输入“女装”并点击，相关的女装列表将会展现给用户，用户点击感兴趣的女装进入详情页，通过查看商品介绍、店家信誉、用户评论等信息综合决定是否购买，如果有M个用户进入同一商品详情页，其中N个购买了该商品，那么该商品的转化率为成交总数和点击总数的比值（N/M）。在这个过程中，如果能够将转化率高的商品返回给用户，那么用户看到的商品正好就是想要购买的商品，这样用户将会更快速地找到喜欢的商品，从而提高用户体验；另一方面，广告每被用户点击一次商家都要付出一定的成本，如果广告被点击却没有成交，广告主将白白付出成本，而如果展现给用户且被点击的广告商品都产生了购买，那么商家虽然付出成本但还是能从成交中获得收益。总结来说，准确预估转化率，能够使得广告主匹配到最可能购买自家商品的用户，提升广告主的投入产出比（ROI）；另一方面，也能让用户快速找到购买意愿最强的商品，从而提升在电商平台中的用户体验。\
   阿里巴巴（淘宝、天猫）是中国最大的电子商务平台，为数亿用户提供了便捷优质的交易服务，也积累了海量的交易数据。阿里妈妈作为阿里巴巴广告业务部门，在过去几年利用这些数据采用深度学习、在线学习、强化学习等人工智能技术来高效准确地预测用户的购买意向，有效提高了用户的购物体验和广告主的ROI。然而，作为一个复杂的生态系统，电商平台中的用户行为偏好、商品长尾分布、热点事件营销等因素依然给转化率预估带来了巨大挑战。比如，在双十一购物狂欢节期间，商家和平台的促销活动会导致流量分布变化剧烈，在正常流量上训练的模型无法很好地匹配这些特殊流量。如何更好地利用海量的交易数据来高效准确地预测用户的购买意向，是人工智能和大数据在电子商务场景中需要继续解决的技术难题。
## 赛题内容
本次比赛以阿里电商广告为研究对象，提供了淘宝平台的海量真实交易数据，参赛选手通过人工智能技术构建预测模型预估用户的购买意向，即给定广告点击相关的用户（user）、广告商品（ad）、检索词（query）、上下文内容（context）、商店（shop）等信息的条件下预测广告产生购买行为的概率（pCVR），形式化定义为：pCVR=P(conversion=1 | query, user, ad, context, shop)。\
结合淘宝平台的业务场景和不同的流量特点，我们定义了以下两类挑战：
（1）日常的转化率预估
（2）特殊日期的转化率预估
## 评估指标
通过logarithmic loss（记为logloss）评估模型效果（越小越好）， 公式如下：
