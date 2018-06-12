#未对程序进行封装，文件读取采用的是绝对地址
#数据预处理——历史转化率
from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
import time
import numpy as np
import sklearn.ensemble 
import lightgbm as lgb
from sklearn.metrics import log_loss
from lightgbm.sklearn import LGBMClassifier
from collections import Counter

InteractiveShell.ast_node_interactivity = 'all'

data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_data_v2.csv')
train_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_train_data_v2.csv')
dev_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_dev_data_v2.csv')
test_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test_data_v2.csv')
test1_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test1_data_v2.csv')
test2_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test2_data_v2.csv')

data.drop(['Unnamed: 0'],axis=1,inplace = True)
train_data.drop(['Unnamed: 0'],axis=1,inplace = True)
dev_data.drop(['Unnamed: 0'],axis=1,inplace = True)
test_data.drop(['Unnamed: 0'],axis=1,inplace = True)
test1_data.drop(['Unnamed: 0'],axis=1,inplace = True)
test2_data.drop(['Unnamed: 0'],axis=1,inplace = True)
all_data = pd.concat([data,test_data])
#-----------------------------------------------item_id历史转化率-----------------------------------------------
def item_id_cvr_f(dataframe,x):
    is_trade_count = len(dataframe.loc[(dataframe.item_id==x.item_id)
                                        &(dataframe.is_trade==1)
                                        &(dataframe.context_timestamp_num<x.context_timestamp_num)])
    not_trade_count = len(dataframe.loc[(dataframe.item_id==x.item_id)
                                        &(dataframe.is_trade==0)
                                        &(dataframe.context_timestamp_num<x.context_timestamp_num)])
    if((is_trade_count==0)&(not_trade_count==0)):
        item_id_cvr_lishi = -1
    else:
        item_id_cvr_lishi = is_trade_count/(is_trade_count+not_trade_count)
    return item_id_cvr_lishi
print('item_id_cvr训练集开始：')
train_data['item_id_cvr_lishi'] = train_data.apply(lambda x:item_id_cvr_f(train_data,x),axis=1)
print('item_id_cvr开发集开始：')
dev_data['item_id_cvr_lishi'] = dev_data.apply(lambda x:item_id_cvr_f(data,x),axis=1)
print('item_id_cvr测试集开始：')
test_data['item_id_cvr_lishi'] = test_data.apply(lambda x:item_id_cvr_f(all_data,x),axis=1)
#-----------------------------------------------item_id_context_id历史转化率-------------------------------------------------
def item_id_context_page_id_cvr_f(dataframe,x):
    is_trade_count = len(dataframe.loc[(dataframe.item_id==x.item_id)
                                       &(dataframe.context_page_id==x.context_page_id)
                                        &(dataframe.is_trade==1)
                                        &(dataframe.context_timestamp_num<x.context_timestamp_num)])
    not_trade_count = len(dataframe.loc[(dataframe.item_id==x.item_id)
                                        &(dataframe.context_page_id==x.context_page_id)
                                        &(dataframe.is_trade==0)
                                        &(dataframe.context_timestamp_num<x.context_timestamp_num)])
    if((is_trade_count==0)&(not_trade_count==0)):
        item_id_context_page_id_cvr_lishi = -1
    else:
        item_id_context_page_id_cvr_lishi = is_trade_count/(is_trade_count+not_trade_count)
    return item_id_context_page_id_cvr_lishi
print('item_id_context_page_id_cvr训练集开始：')
train_data['item_id_context_page_cvr_lishi'] = train_data.apply(lambda x:item_id_context_page_id_cvr_f(train_data,x),axis=1)
print('item_id_context_page_id_cvr开发集开始：')
dev_data['item_id_context_page_cvr_lishi'] = dev_data.apply(lambda x:item_id_context_page_id_cvr_f(data,x),axis=1)
print('item_id_context_page_id_cvr测试集开始：')
test_data['item_id_context_page_cvr_lishi'] = test_data.apply(lambda x:item_id_context_page_id_cvr_f(all_data,x),axis=1)
#------------------------------------------------item_id_user_gender历史转化率-------------------------------------------------
def item_id_user_gender_id_cvr_f(dataframe,x):
    is_trade_count = len(dataframe.loc[(dataframe.item_id==x.item_id)
                                       &(dataframe.user_gender_id==x.user_gender_id)
                                        &(dataframe.is_trade==1)
                                        &(dataframe.context_timestamp_num<x.context_timestamp_num)])
    not_trade_count = len(dataframe.loc[(dataframe.item_id==x.item_id)
                                        &(dataframe.user_gender_id==x.user_gender_id)
                                        &(dataframe.is_trade==0)
                                        &(dataframe.context_timestamp_num<x.context_timestamp_num)])
    if((is_trade_count==0)&(not_trade_count==0)):
        item_id_user_gender_id_cvr_lishi = -1
    else:
        item_id_user_gender_id_cvr_lishi = is_trade_count/(is_trade_count+not_trade_count)
    return item_id_user_gender_id_cvr_lishi
print('item_id_user_gender_id_cvr训练集开始：')
train_data['item_id_user_gender_cvr_lishi'] = train_data.apply(lambda x:item_id_user_gender_id_cvr_f(train_data,x),axis=1)
print('item_id_user_gender_id_cvr开发集开始：')
dev_data['item_id_user_gender_cvr_lishi'] = dev_data.apply(lambda x:item_id_user_gender_id_cvr_f(data,x),axis=1)
print('item_id_user_gender_id_cvr测试集开始：')
test_data['item_id_user_gender_cvr_lishi'] = test_data.apply(lambda x:item_id_user_gender_id_cvr_f(all_data,x),axis=1)
#--------------------------------------------------item_id_user_age历史转化率------------------------------------------------
def item_id_user_age_level_cvr_f(dataframe,x):
    is_trade_count = len(dataframe.loc[(dataframe.item_id==x.item_id)
                                       &(dataframe.user_age_level==x.user_age_level)
                                        &(dataframe.is_trade==1)
                                        &(dataframe.context_timestamp_num<x.context_timestamp_num)])
    not_trade_count = len(dataframe.loc[(dataframe.item_id==x.item_id)
                                        &(dataframe.user_age_level==x.user_age_level)
                                        &(dataframe.is_trade==0)
                                        &(dataframe.context_timestamp_num<x.context_timestamp_num)])
    if((is_trade_count==0)&(not_trade_count==0)):
        item_id_user_age_level_cvr_lishi = -1
    else:
        item_id_user_age_level_cvr_lishi = is_trade_count/(is_trade_count+not_trade_count)
    return item_id_user_age_level_cvr_lishi
print('item_id_user_age_level_cvr训练集开始：')
train_data['item_id_user_age_cvrlishi'] = train_data.apply(lambda x:item_id_user_age_level_cvr_f(train_data,x),axis=1)
print('item_id_user_age_level_cvr开发集开始：')
dev_data['item_id_user_age_cvrlishi'] = dev_data.apply(lambda x:item_id_user_age_level_cvr_f(data,x),axis=1)
print('item_id_user_age_level_cvr测试集开始：')
test_data['item_id_user_age_cvrlishi'] = test_data.apply(lambda x:item_id_user_age_level_cvr_f(all_data,x),axis=1)

test1_data_pre,test2_data_pre = test_data[:test1_data.shape[0]],test_data[test1_data.shape[0]:]
data_all = pd.concat([train_data,dev_data])
data_all.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_data_v3.csv')
train_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_train_data_v3.csv')
dev_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_dev_data_v3.csv')
test_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test_data_v3.csv')
test1_data_pre.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test1_data_v3.csv')
#test2_data为初赛二阶段提交的data
test2_data_pre.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test2_data_v3.csv')

