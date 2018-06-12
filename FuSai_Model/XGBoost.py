#复赛XGBoost单模型
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
from xgboost.sklearn import XGBClassifier
import columns

InteractiveShell.ast_node_interactivity = 'all'

cv = StratifiedKFold(n_splits=5,random_state=100)

selected_columns=columns.selected_columns

def output(content):
    f = open('model_xgboost_log.txt','a+')
    f.write(content)
    f.write('\n')
    f.close()

#201
train_data = pd.read_csv('/home/share/liangxiao/second_stage/feature_folder/feature_v9/data_7th_up_v9.csv')
test_data = pd.read_csv('/home/share/liangxiao/second_stage/feature_folder/feature_v9/data_7th_down_2_v9.csv')

print(train_data.columns)

train_data['item_category_list_third']=train_data['item_category_list_third'].fillna(-1)
test_data['item_category_list_third']=test_data['item_category_list_third'].fillna(-1)

submit_df_xgboost = pd.DataFrame(np.zeros((len(test_data.index),2)), columns = ['instance_id','predicted_score'])
submit_df_xgboost['instance_id'] = test_data['instance_id']

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
X_test = test_data.as_matrix()


i=0
score=0
for train_index,dev_index in cv.split(X_train,y_train):
    i=i+1
    X_train_set=X_train.iloc[train_index].as_matrix()
    y_train_set=y_train.iloc[train_index].as_matrix()
    X_dev_set=X_train.iloc[dev_index].as_matrix()
    y_dev_set=y_train.iloc[dev_index].as_matrix()
    xgb1 = XGBClassifier(learning_rate=0.03,
        n_estimators=1000,
        max_depth=7,
        colsample_bytree=0.8,
        objective = 'binary:logistic',
        seed=2048)
    xgb1.fit(X_train_set,y_train_set)
    predict_dev=xgb1.predict_proba(X_dev_set)[:,1]
    print(predict_dev)
    score_each=log_loss(y_dev_set,predict_dev)
    output('Each log_loss: %f'%score_each)
    score = score+score_each
    submit_df_xgboost['predicted_score_%s'%i]=xgb1.predict_proba(X_test)[:,1]
output('Xgboost Total Log_loss: %f'%(score/5))
submit_df_xgboost['predicted_score'] = (submit_df_xgboost['predicted_score_1']+submit_df_xgboost['predicted_score_2']
                                +submit_df_xgboost['predicted_score_3']+submit_df_xgboost['predicted_score_4']
                                +submit_df_xgboost['predicted_score_5'])/5
submit_df_xgboost = submit_df_xgboost[['instance_id','predicted_score']]
submit_df_xgboost.to_csv('/home/share/liangxiao/submit_folder/submit_data_514_lx_xgboost_new_2.txt', sep=" ", index=False, line_terminator='\n')
