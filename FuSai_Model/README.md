模型文件说明
====
# 单模型
复赛我主要采用了三种单模型，XGBoost、LightGBM、LightGBM-DART
# 模型融合思路
我和一个队友（老王）分别用各自的特征跑单模型，然后分别Average，另外一个队友利用我们的单模型和特征构建两层的Stacking（单模型+LR），
最后把两个Average和一个Stacking的结果再做一次Average，就是我们的最终结果（之后补流程图）
