#未对程序进行封装，文件读取用的是绝对路径
#统计信息——时间差
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
#数据读取部分
data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_data_v4.csv')
train_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_train_data_v4.csv')
dev_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_dev_data_v4.csv')
test_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test_data_v4.csv')
test1_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test1_data_v4.csv')
test2_data = pd.read_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test2_data_v4.csv')

data.drop(['Unnamed: 0'],axis=1,inplace = True)
train_data.drop(['Unnamed: 0'],axis=1,inplace = True)
dev_data.drop(['Unnamed: 0'],axis=1,inplace = True)
test_data.drop(['Unnamed: 0'],axis=1,inplace = True)
test1_data.drop(['Unnamed: 0'],axis=1,inplace = True)
test2_data.drop(['Unnamed: 0'],axis=1,inplace = True)
all_data = pd.concat([data,test_data])
#*****************************用户本次浏览商品和上一次浏览商品的时间差*****************************************************
def item_user_shijiancha_f(dataframe,x):
    if(x.item_id_user_id_lishi==0):
        itemid_userid_shijiancha = -1
    else:
        itemid_userid_shijian = list(dataframe.loc[(dataframe.item_id==x.item_id)
                                                          &(dataframe.user_id==x.user_id)
                                                          &(dataframe.context_timestamp_num<=x.context_timestamp_num)].sort_values(ascending=False,by=['context_timestamp_num'])['context_timestamp_num'])
        itemid_userid_shijiancha = itemid_userid_shijian[0]-itemid_userid_shijian[1]
    return itemid_userid_shijiancha
print('商品-用户与上一次浏览时间差（训练集）开始：')
train_data['item_id_user_id_shijiancha'] = train_data.apply(lambda x:item_user_shijiancha_f(all_data,x),axis=1)
print('商品-用户与上一次浏览时间差（开发集）开始：')
dev_data['item_id_user_id_shijiancha'] = dev_data.apply(lambda x:item_user_shijiancha_f(all_data,x),axis=1)
print('商品-用户与上一次浏览时间差（测试集）开始：')
test_data['item_id_user_id_shijiancha'] = test_data.apply(lambda x:item_user_shijiancha_f(all_data,x),axis=1)
#*****************************用户本次浏览二次类目和上一次浏览二级类目的时间差*********************************************
def categorysecond_user_shijiancha_f(dataframe,x):
    if(x.category_second_user_id_lishi==0):
        categorysecond_userid_shijiancha = -1
    else:
        categorysecond_userid_shijian = list(dataframe.loc[(dataframe.item_category_list_second==x.item_category_list_second)
                                                          &(dataframe.user_id==x.user_id)
                                                          &(dataframe.context_timestamp_num<=x.context_timestamp_num)].sort_values(ascending=False,by=['context_timestamp_num'])['context_timestamp_num'])
        categorysecond_userid_shijiancha = categorysecond_userid_shijian[0]-categorysecond_userid_shijian[1]
    return categorysecond_userid_shijiancha
print('二级类目-用户与上一次浏览时间差（训练集）开始：')
train_data['category_second_user_id_shijiancha'] = train_data.apply(lambda x:categorysecond_user_shijiancha_f(all_data,x),axis=1)
print('二级类目-用户与上一次浏览时间差（开发集）开始：')
dev_data['category_second_user_id_shijiancha'] = dev_data.apply(lambda x:categorysecond_user_shijiancha_f(all_data,x),axis=1)
print('二级类目-用户与上一次浏览时间差（测试集）开始：')
test_data['category_second_user_id_shijiancha'] = test_data.apply(lambda x:categorysecond_user_shijiancha_f(all_data,x),axis=1)

#下面的两个特征都用到了leak，即用到了与下一次浏览之前的时间差
#******************************用户本次浏览商品和下一次浏览商品的时间差*************************************
def item_user_shijiancha_leak_f(dataframe,x):
    itemid_userid_series_leak = dataframe.loc[(dataframe.user_id==x.user_id)
                                                   &(dataframe.item_id==x.item_id)
                                                   &(dataframe.context_timestamp_num>=x.context_timestamp_num)]
    itemid_userid_length_leak = len(itemid_userid_series_leak)
    if(itemid_userid_length_leak==1):
        itemid_userid_shijiancha_leak = -1
    else:
        itemid_userid_shijian_leak = list(itemid_userid_series_leak.sort_values(by=['context_timestamp_num'])['context_timestamp_num'])
        itemid_userid_shijiancha_leak = itemid_userid_shijian_leak[1]-itemid_userid_shijian_leak[0]
    return itemid_userid_shijiancha_leak
print('商品-用户与下一次浏览时间差（训练集）开始：')
train_data['item_id_user_id_shijiancha_leak'] = train_data.apply(lambda x:item_user_shijiancha_leak_f(all_data,x),axis=1)
print('商品-用户与下一次浏览时间差（开发集）开始：')
dev_data['item_id_user_id_shijiancha_leak'] = dev_data.apply(lambda x:item_user_shijiancha_leak_f(all_data,x),axis=1)
print('商品-用户与下一次浏览时间差（测试集）开始：')
test_data['item_id_user_id_shijiancha_leak'] = test_data.apply(lambda x:item_user_shijiancha_leak_f(all_data,x),axis=1)
#******************************用户本次浏览二级类目和下一次浏览二级类目的时间差*******************************************
def categorysecond_user_shijiancha_leak_f(dataframe,x):
    categorysecond_userid_series_leak = dataframe.loc[(dataframe.user_id==x.user_id)
                                                           &(dataframe.item_category_list_second==x.item_category_list_second)
                                                           &(dataframe.context_timestamp_num>=x.context_timestamp_num)]
    categorysecond_userid_length_leak = len(categorysecond_userid_series_leak)
    if(categorysecond_userid_length_leak==1):
        categorysecond_userid_shijiancha_leak = -1
    else:
        categorysecond_userid_shijian_leak = list(categorysecond_userid_series_leak.sort_values(by=['context_timestamp_num'])['context_timestamp_num'])
        categorysecond_userid_shijiancha_leak = categorysecond_userid_shijian_leak[1]-categorysecond_userid_shijian_leak[0]
    return categorysecond_userid_shijiancha_leak
print('二级类目-用户与下一次浏览时间差（训练集）开始：')
train_data['category_second_user_id_shijiancha_leak'] = train_data.apply(lambda x:categorysecond_user_shijiancha_leak_f(all_data,x),axis=1)
print('二级类目-用户与下一次浏览时间差（开发集）开始：')
dev_data['category_second_user_id_shijiancha_leak'] = dev_data.apply(lambda x:categorysecond_user_shijiancha_leak_f(all_data,x),axis=1)
print('二级类目-用户与下一次浏览时间差（测试集）开始：')
test_data['category_second_user_id_shijiancha_leak'] = test_data.apply(lambda x:categorysecond_user_shijiancha_leak_f(all_data,x),axis=1)

#从全局考虑，用户第一次浏览到最后一次浏览之间的时间差，这里也用到了leak
#******************************用户第一次浏览商品和最后一次浏览商品的时间差***********************************
def item_user_shijiancha_all_f(dataframe,x):
    if(x['user_id_item_id_all']==1):
        userid_itemid_shijiancha_all = 0
    else:
        userid_itemid_shijian_all = list(dataframe.loc[(dataframe.item_id==x.item_id)
                                                          &(dataframe.user_id==x.user_id)].sort_values(ascending=False,by=['context_timestamp_num'])['context_timestamp_num'])
        userid_itemid_shijiancha_all = userid_itemid_shijian_all[0]-userid_itemid_shijian_all[-1]
    return userid_itemid_shijiancha_all
print('商品-用户与下一次浏览时间差（训练集）开始：')
train_data['user_id_item_id_shijiancha_all'] = train_data.apply(lambda x:item_user_shijiancha_all_f(all_data,x),axis=1)
print('商品-用户与下一次浏览时间差（开发集）开始：')
dev_data['user_id_item_id_shijiancha_all'] = dev_data.apply(lambda x:item_user_shijiancha_all_f(all_data,x),axis=1)
print('商品-用户与下一次浏览时间差（测试集）开始：')
test_data['user_id_item_id_shijiancha_all'] = test_data.apply(lambda x:item_user_shijiancha_all_f(all_data,x),axis=1)
#******************************用户第一次浏览二级类目和最后一次浏览二级类目的时间差**************************************
def categorysecond_user_shijiancha_all_f(dataframe,x):
    if(x['user_id_category_second_all']==1):
        userid_categorysecond_shijiancha_all = 0
    else:
        userid_categorysecond_shijian_all = list(dataframe.loc[(dataframe.item_category_list_second==x.item_category_list_second)
                                                                 &(dataframe.user_id==x.user_id)].sort_values(ascending=False,by=['context_timestamp_num'])['context_timestamp_num'])
        userid_categorysecond_shijiancha_all = userid_categorysecond_shijian_all[0]-userid_categorysecond_shijian_all[-1]
    return userid_categorysecond_shijiancha_all
print('二级类目-用户与下一次浏览时间差（训练集）开始：')
train_data['user_id_category_second_shijiancha_all'] = train_data.apply(lambda x:categorysecond_user_shijiancha_all_f(all_data,x),axis=1)
print('二级类目-用户与下一次浏览时间差（开发集）开始：')
dev_data['user_id_category_second_shijiancha_all'] = dev_data.apply(lambda x:categorysecond_user_shijiancha_all_f(all_data,x),axis=1)
print('二级类目-用户与下一次浏览时间差（测试集）开始：')
test_data['user_id_category_second_shijiancha_all'] = test_data.apply(lambda x:categorysecond_user_shijiancha_all_f(all_data,x),axis=1)


test1_data_pre,test2_data_pre = test_data[:test1_data.shape[0]],test_data[test1_data.shape[0]:]
data_all = pd.concat([train_data,dev_data])
data_all.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_data_v5.csv')
train_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_train_data_v5.csv')
dev_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_dev_data_v5.csv')
test_data.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test_data_v5.csv')
test1_data_pre.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test1_data_v5.csv')
#test2_data为初赛二阶段提交的data
test2_data_pre.to_csv('G:/Python/Alimama/Chusai/preprocessing_second_stage_test2_data_v5.csv')
