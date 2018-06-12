import numpy as np
import pandas as pd

import lightgbm as lgb
from sklearn import metrics, preprocessing

from no_valid_oneHot import get_no_valid_feature



class greedyFeatureSelection(object):
    
    def __init__(self, Xtrain, ytrain, Xtest, ytest, scale=1, verbose=1):
        
        data = pd.concat([Xtrain,Xtest])
        
        if scale == 1:
            self._data = preprocessing.scale(np.array(data))
        else:
            self._data = np.array(data)
        
        self._Xtrain = self._data[0:len(Xtrain),]
        self._ytrain = ytrain
        self._Xtest = self._data[len(Xtrain):,]
        self._ytest = ytest
        
        self._verbose = verbose
    
    def evaluateScore(self, Xts_train, Xts_test):
        model = lgb.LGBMClassifier(num_leaves=24, max_depth=5, n_estimators=100, n_jobs=-1)
        model.fit(Xts_train, self._ytrain)
        
        predictions = model.predict_proba(Xts_test)[:, 1]
        score = metrics.log_loss(self._ytest, predictions)
        return score
	
	
    def selectionLoop(self):
        score_history = []
        good_features = set([])
        num_features = self._Xtrain.shape[1]
        while len(score_history) < 2 or score_history[-1][0] < score_history[-2][0]:
            scores = []
            for feature in range(num_features):
                if feature not in good_features:
                    selected_features = list(good_features) + [feature]
                
                    Xts_train = np.column_stack(self._Xtrain[:, j] for j in selected_features)
                    Xts_test = np.column_stack(self._Xtest[:, j] for j in selected_features)
                    
                    score = self.evaluateScore(Xts_train, Xts_test)
                    scores.append((score, feature))
                    
                    if self._verbose:
                        print(feature)
                        print("Current Logloss : ", np.mean(score))
                        f = open('logistic_log_wang.txt','a+')
                        f.write("Current Logloss : "+str(np.mean(score))+'\n')
                        f.close()
        	
            good_features.add(sorted(scores)[0][1])
            score_history.append(sorted(scores)[0])
            if self._verbose:
                f = open('logistic_log_wang.txt','a+')
                print('-----------------------------------------------')
                f.write("-----------------------------------------------\n")
                print(sorted(scores)[0][1])
                f.write(str(sorted(scores)[0][1])+"\n")
                print("Current Features : ", sorted(list(good_features)))
                f.write("Current Features : "+str(sorted(list(good_features)))+'\n')
                print("Each Step Logloss : ", score_history[-1][0])
                f.write("Each Step Logloss : "+str(score_history[-1][0])+'\n')
                print('-----------------------------------------------')
                f.write("-----------------------------------------------\n ")
                f.close()
               
		
        # Remove last added feature
        good_features.remove(score_history[-1][1])
        good_features = sorted(list(good_features))
        if self._verbose:
            f = open('logistic_log_wang.txt','a+')
            print("Selected Features : ", good_features)
            for element in good_features:
                f.write(str(element)+'\t')
            f.write('\n')
            f.close()
        
        return good_features
    
    def transform(self):
        good_features = self.selectionLoop()
       
        return good_features



path_my = '/home/share/wangtiefei/'

#print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))


data_all = pd.read_csv(path_my+"data/data_all_add_sort_v1.csv", encoding='utf8')


d_train = data_all[data_all.day != 7]
d_eval = data_all[(data_all['is_trade'] != 999) & (data_all.day == 7)]

label = 'is_trade'
no_valid_feature = get_no_valid_feature()
features = [x for x in d_train.columns if x not in no_valid_feature]


gf = greedyFeatureSelection(d_train[features],d_train['is_trade'],
                            d_eval[features],d_eval['is_trade'],
                            scale=0,verbose=1)


gf.transform()
