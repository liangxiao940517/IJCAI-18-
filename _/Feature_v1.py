#数据预处理
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time

from sklearn.ensemble import GradientBoostingClassifier
#读取原始文件
data = pd.read_csv('G:/Python/Alimama/round1_ijcai_18_train_20180301.txt', sep =' ')
test1_data = pd.read_csv('G:/Python/Alimama/round1_ijcai_18_test_a_20180301.txt', sep = ' ')
test2_data = pd.read_csv('G:/Python/Alimama/round1_ijcai_18_test_b_20180418.txt', sep = ' ')
test_data = pd.concat([test1_data,test2_data]).reset_index()
# data['item_sales_level'].value_counts()

#将context_page_id映射为1-20
map_context_page_id = {4001:1,4002:2,4003:3,4004:4,4005:5,4006:6,4007:7,4008:8,4009:9,4010:10,4011:11,4012:12,4013:13,4014:14,4015:15,4016:16,4017:17,4018:18,4019:19,4020:20}
data['context_page_id'] = data['context_page_id'].map(map_context_page_id)
test_data['context_page_id'] = test_data['context_page_id'].map(map_context_page_id)

#将shop_star_level映射为1-22
map_shop_star_level = {4999:1,5000:2,5001:3,5002:4,5003:5,5004:6,5005:7,5006:8,5007:9,5008:10,5009:11,5010:12,5011:13,5012:14,5013:15,5014:16,5015:17,5016:18,5017:19,5018:20,5019:21,5020:22}
data['shop_star_level'] = data['shop_star_level'].map(map_shop_star_level)
test_data['shop_star_level'] = test_data['shop_star_level'].map(map_shop_star_level)
#将user_age_level映射为-1-7
map_user_age_level = {1000:0,1001:1,1002:2,1003:3,1004:4,1005:5,1006:6,1007:7,-1:-1}
data['user_age_level'] = data['user_age_level'].map(map_user_age_level)
test_data['user_age_level'] = test_data['user_age_level'].map(map_user_age_level)
#将user_star_level映射为-1-10
map_user_star_level = {3000:0,3001:1,3002:2,3003:3,3004:4,3005:5,3006:6,3007:7,3008:8,3009:9,3010:10,-1:-1}
data['user_star_level'] = data['user_star_level'].map(map_user_star_level)
test_data['user_star_level'] = test_data['user_star_level'].map(map_user_star_level)

#填补shop_score_service缺失值，平均数
mean_shop_score_service = data[data['shop_score_service']!=-1]['shop_score_service'].mean()
missing_shop_score_service_index = data[data['shop_score_service']==-1].index
data.loc[missing_shop_score_service_index,'shop_score_service'] = mean_shop_score_service
missing_shop_score_service_index_test = test_data[test_data['shop_score_service']==-1].index
test_data.loc[missing_shop_score_service_index_test,'shop_score_service'] = mean_shop_score_service

#填补shop_score_delivery缺失值，平均数
mean_shop_score_delivery = data[data['shop_score_delivery']!=-1]['shop_score_delivery'].mean()
missing_shop_score_delivery_index = data[data['shop_score_delivery']==-1].index
data.loc[missing_shop_score_delivery_index,'shop_score_delivery'] = mean_shop_score_delivery
missing_shop_score_delivery_index_test = test_data[test_data['shop_score_delivery']==-1].index
test_data.loc[missing_shop_score_delivery_index_test,'shop_score_delivery'] = mean_shop_score_delivery
#填补shop_score_description缺失值，平均数
mean_shop_score_description = data[data['shop_score_description']!=-1]['shop_score_description'].mean()
missing_shop_score_description_index = data[data['shop_score_description']==-1].index
data.loc[missing_shop_score_description_index,'shop_score_description'] = mean_shop_score_description
missing_shop_score_description_index_test = test_data[test_data['shop_score_description']==-1].index
test_data.loc[missing_shop_score_description_index_test,'shop_score_description'] = mean_shop_score_description
#填补shop_review_positive_rate缺失值，平均数
mean_shop_review_positive_rate = data[data['shop_review_positive_rate']!=-1]['shop_review_positive_rate'].mean()
missing_shop_review_positive_rate_index = data[data['shop_review_positive_rate']==-1].index
data.loc[missing_shop_review_positive_rate_index, 'shop_review_positive_rate'] = mean_shop_review_positive_rate
missing_shop_review_positive_rate_index_test = test_data[test_data['shop_review_positive_rate']==-1].index
test_data.loc[missing_shop_review_positive_rate_index_test,'shop_review_positive_rate'] = mean_shop_review_positive_rate

#填补item_sales_level缺失值,相当于用item的其他属性训练模型来填补sales的缺失值，初赛缺失值只有900多，而且数据量不大，
#可以考虑用lgb速度快一点，也可以考虑不处理，不处理的话注释掉下面一段代码即可。
item_sales_level_missing = data[data['item_sales_level']==-1].loc[:,['item_brand_id','item_city_id','item_price_level','item_collected_level','item_pv_level']]
missing_item_sales_level_index = item_sales_level_missing.index
item_sales_level_notmissing = data[data['item_sales_level']!=-1].loc[:,['item_brand_id','item_city_id','item_price_level','item_collected_level','item_pv_level','item_sales_level']]
X_train_sales = item_sales_level_notmissing.iloc[:,0:5].as_matrix()
y_train_sales = item_sales_level_notmissing.iloc[:,5].as_matrix()
X_predict_sales = item_sales_level_missing.as_matrix()
gbc = GradientBoostingClassifier(n_estimators = 100, learning_rate = 0.1, random_state = 2048)
print('Start Training!')
gbc.fit(X_train_sales, y_train_sales)
print('Finish Training!')
y_predict_sales = gbc.predict(X_predict_sales)
data.loc[missing_item_sales_level_index,'item_sales_level'] = y_predict_sales
data['item_sales_level'].value_counts()
item_sales_level_missing_test = test_data[test_data['item_sales_level']==-1].loc[:,['item_brand_id','item_city_id','item_price_level','item_collected_level','item_pv_level']]
missing_item_sales_level_index_test = item_sales_level_missing_test.index
X_predict_sales_test = item_sales_level_missing_test.as_matrix()
print('Predict test sets!')
y_predict_sales_test = gbc.predict(X_predict_sales_test)
print('Finished!')
test_data.loc[missing_item_sales_level_index_test,'item_sales_level'] = y_predict_sales_test


#衍生特征
data['shop_score_whole'] = data['shop_score_service'] + data['shop_score_delivery'] + data['shop_score_description']
data['shop_review_positive_level'] = data['shop_review_num_level'] * data['shop_review_positive_rate']
data['item_collected_rate'] = data['item_collected_level']/data['item_pv_level']
data['item_income_level'] = data['item_price_level'] * data['item_sales_level']

test_data['shop_score_whole'] = test_data['shop_score_service'] + test_data['shop_score_delivery'] + test_data['shop_score_description']
test_data['shop_review_positive_level'] = test_data['shop_review_num_level'] * test_data['shop_review_positive_rate']
test_data['item_collected_rate'] = test_data['item_collected_level']/test_data['item_pv_level']
test_data['item_income_level'] = test_data['item_price_level'] * test_data['item_sales_level']

# 处理item_category_list
item_category = data['item_category_list'].str.split(';',expand = True)
item_category.columns = ['item_category_list_first','item_category_list_second','item_category_list_third']
data = pd.concat([data,item_category], axis = 1)

item_category_test = test_data['item_category_list'].str.split(';',expand = True)
item_category_test.columns = ['item_category_list_first','item_category_list_second','item_category_list_third']
test_data = pd.concat([test_data,item_category_test], axis = 1)

# 对于context_timestamp的处理，按时间划分训练集与开发集
def convert_context_timestamp_num(columns):
    format = '%Y%m%d%H%M%S'
    value = time.localtime(columns)
    dt = time.strftime(format, value)
    return dt
data['context_timestamp_num'] = data['context_timestamp'].apply(convert_context_timestamp_num)
test_data['context_timestamp_num'] = test_data['context_timestamp'].apply(convert_context_timestamp_num)


# 对于context_timestamp的处理，按时间划分训练集与开发集
def convert_context_timestamp(columns):
    format = '%Y %m %d %H %M %S'
    value = time.localtime(columns)
    dt = time.strftime(format, value)
    return dt
data['context_timestamp'] = data['context_timestamp'].apply(convert_context_timestamp)
# data['context_timestamp']
timestamp = data['context_timestamp'].str.split(' ',expand = True)
timestamp.columns = ['Year','Month','Day','Hour','Minute','Second']
data = pd.concat([data, timestamp], axis = 1)

test_data['context_timestamp'] = test_data['context_timestamp'].apply(convert_context_timestamp)
# test_data['context_timestamp']
timestamp_test = test_data['context_timestamp'].str.split(' ',expand = True)
timestamp_test.columns = ['Year','Month','Day','Hour','Minute','Second']
test_data = pd.concat([test_data, timestamp_test], axis = 1)

all_data = pd.concat([data,test_data])

train_data = data[data['Day'].isin(['18','19','20','21','22','23'])]
dev_data = data[data['Day'].isin(['24'])]
test1_data,test2_data = test_data[:test1_data.shape[0]],test_data[test1_data.shape[0]:]

#生成v1版数据，未开放B榜前，注释掉test1_data和test2_data即可
data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_data_v1.csv')
train_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_train_data_v1.csv')
dev_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_dev_data_v1.csv')
test_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test_data_v1.csv')
test1_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test1_data_v1.csv')
#test2_data为初赛二阶段提交的data
test2_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test2_data_v1.csv')
