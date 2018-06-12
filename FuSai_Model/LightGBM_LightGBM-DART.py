#lgb和lgb-dart
from IPython.core.interactiveshell import InteractiveShell
import pandas as pd
import time
import numpy as np
import sklearn.ensemble 
import lightgbm as lgb
from sklearn.metrics import log_loss
from lightgbm.sklearn import LGBMClassifier
from collections import Counter
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
import columns

InteractiveShell.ast_node_interactivity = 'all'

cv = StratifiedKFold(n_splits=5,random_state=100)

selected_columns=columns.selected_columns

#201
train_data = pd.read_csv('/home/share/liangxiao/second_stage/feature_folder/feature_v9/data_7th_up_v9.csv')
test_data = pd.read_csv('/home/share/liangxiao/second_stage/feature_folder/feature_v9/data_7th_down_2_v9.csv')

print(train_data.columns)

train_data['item_category_list_third']=train_data['item_category_list_third'].fillna(-1)
test_data['item_category_list_third']=test_data['item_category_list_third'].fillna(-1)

submit_df_lgb = pd.DataFrame(np.zeros((len(test_data.index),2)), columns = ['instance_id','predicted_score'])
submit_df_dart = pd.DataFrame(np.zeros((len(test_data.index),2)), columns = ['instance_id','predicted_score'])
submit_df_lgb['instance_id'] = test_data['instance_id']
submit_df_dart['instance_id'] = test_data['instance_id']

train_data_y = train_data['is_trade']

train_data = train_data.loc[:,selected_columns]
test_data = test_data.loc[:,selected_columns]

train_data_feature = train_data
#X_train = train_data_feature.as_matrix()
X_train = train_data_feature
train_data_label = train_data_y
#y_train = train_data_label.as_matrix()
y_train = train_data_label


#X_test = test_data.as_matrix()
X_test = test_data


i=0
score=0
for train_index,dev_index in cv.split(X_train,y_train):
    i=i+1
    X_train_set=X_train.iloc[train_index].as_matrix()
    y_train_set=y_train.iloc[train_index].as_matrix()
    X_dev_set=X_train.iloc[dev_index].as_matrix()
    y_dev_set=y_train.iloc[dev_index].as_matrix()
    lgb_train_set=lgb.Dataset(X_train_set,label=y_train_set)
    lgb_params = {'task':'train',
             'boosting':'gbdt',
             'application':'binary',
             'meric':'binary_logloss',
              'num_leaves':63,
              'max_depth':7,
#               'min_data_in_leaf':100,
             'learning_rate':0.03,
             'feature_fraction':0.8,
             'bagging_fraction':1.0,
             'bagging_freq':1,
             'bagging_seed':1024,
             'feature_fraction_seed':1024
             }
    bst=lgb.train(lgb_params,lgb_train_set,num_boost_round=2000)
    predict_dev=bst.predict(X_dev_set,raw_score=False)
    print(predict_dev)
    score_each=log_loss(y_dev_set,predict_dev)
    print('Each log_loss: %f'%score_each)
    score = score+score_each
    submit_df_lgb['predicted_score_%s'%i]=bst.predict(X_test,raw_score = False)
print('Lightgbm Total Log_loss: %f'%(score/5))
submit_df_lgb['predicted_score'] = (submit_df_lgb['predicted_score_1']+submit_df_lgb['predicted_score_2']
                                +submit_df_lgb['predicted_score_3']+submit_df_lgb['predicted_score_4']
                                +submit_df_lgb['predicted_score_5'])/5
submit_df_lgb = submit_df_lgb[['instance_id','predicted_score']]
submit_df_lgb.to_csv('/home/share/liangxiao/submit_folder/submit_data_513_lx_lgb.txt', sep=" ", index=False, line_terminator='\n')




i=0
score=0
for train_index,dev_index in cv.split(X_train,y_train):
    i=i+1
    X_train_set=X_train.iloc[train_index].as_matrix()
    y_train_set=y_train.iloc[train_index].as_matrix()
    X_dev_set=X_train.iloc[dev_index].as_matrix()
    y_dev_set=y_train.iloc[dev_index].as_matrix()
    lgb_train_set=lgb.Dataset(X_train_set,label=y_train_set)
    lgb_params_dart = {
            'boosting': 'dart',
            'objective': 'binary',
            'metric': 'binary_logloss',
            'learning_rate': 0.1,
            'num_leaves': 32,
            'max_depth': 7,
            'bagging_fraction': 0.8,
            'drop_rate': 0.1,
            'uniform_drop': 'true',
            'colsample_bytree': 0.3,
            'xgboost_dart_mode': 'true'
        }
    bst_dart=lgb.train(lgb_params_dart,lgb_train_set,num_boost_round=230)
    predict_dev_dart=bst_dart.predict(X_dev_set,raw_score=False)
    print(predict_dev_dart)
    score_each=log_loss(y_dev_set,predict_dev_dart)
    print('Each log_loss: %f'%score_each)
    score = score+score_each
    submit_df_dart['predicted_score_%s'%i]=bst_dart.predict(X_test,raw_score = False)
print('Lightgbm Dart Total Log_loss: %f'%(score/5))
submit_df_dart['predicted_score'] = (submit_df_dart['predicted_score_1']+submit_df_dart['predicted_score_2']
                                +submit_df_dart['predicted_score_3']+submit_df_dart['predicted_score_4']
                                +submit_df_dart['predicted_score_5'])/5
submit_df_dart = submit_df_lgb[['instance_id','predicted_score']]
submit_df_dart.to_csv('/home/share/liangxiao/submit_folder/submit_data_513_lx_lgb_dart.txt', sep=" ", index=False, line_terminator='\n')
