from MLFeatureSelection import FeatureSelection as FS
from sklearn.metrics import log_loss
import lightgbm as lgbm
import pandas as pd
import numpy as np
import columns_2
from sklearn.model_selection import KFold,StratifiedKFold,ShuffleSplit

def prepareData():
    df = pd.read_csv('/home/share/liangxiao/second_stage/feature_selection/cv_folder/train_data.csv')
    df = df[~pd.isnull(df.is_trade)]
    item_category_list_unique = list(np.unique(df.item_category_list))
    df.item_category_list.replace(item_category_list_unique, list(np.arange(len(item_category_list_unique))), inplace=True)
    return df

def modelscore(y_test, y_pred):
    return log_loss(y_test, y_pred)

#def validation(X,y,features, clf,lossfunction):
#    totaltest = 0
#    for D in [10,11]:
#        T = (X.Hour != D)
#        X_train, X_test = X[T], X[~T]
#        X_train, X_test = X_train[features], X_test[features]
#        y_train, y_test = y[T], y[~T]
#        clf.fit(X_train,y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=200)
#        totaltest += lossfunction(y_test, clf.predict_proba(X_test)[:,1])
#    totaltest /= 1.0
#    return totaltest

def validation(X,y, features, clf, lossfunction):
    totaltest = []
    kf = StratifiedKFold(5,random_state=100)
    for train_index, test_index in kf.split(X,y):
        X_train, X_test = X.ix[train_index,:][features], X.ix[test_index,:][features]
        y_train, y_test = y[train_index], y[test_index]
        #clf.fit(X_train, y_train, eval_set = [(X_train, y_train), (X_test, y_test)], eval_metric='logloss', verbose=False,early_stopping_rounds=50)
        clf.fit(X_train, y_train)
        totaltest.append(lossfunction(y_test, clf.predict(X_test)))
    return np.mean(totaltest)

def add(x,y):
    return x + y

def substract(x,y):
    return x - y

def times(x,y):
    return x * y

def divide(x,y):
    return (x + 0.001)/(y + 0.001)

CrossMethod = {'+':add,
               '-':substract,
               '*':times,
               '/':divide,}

def main():
    sf = FS.Select(Sequence = True, Random = False, Cross = False) #select the way you want to process searching
    sf.ImportDF(prepareData(),label = 'is_trade')
    sf.ImportLossFunction(modelscore,direction = 'descend')
    sf.ImportCrossMethod(CrossMethod)
    sf.InitialNonTrainableFeatures(columns_2.drop_columns)
    sf.InitialFeatures([])
    sf.GenerateCol(key = 'mean', selectstep = 2)
    sf.SetSample(0.1, samplemode = 0, samplestate = 0)
    sf.AddPotentialFeatures(columns_2.all_select_columns)
#    sf.SetFeaturesLimit(20)
    sf.SetTimeLimit(100)
    sf.clf = lgbm.LGBMClassifier(random_state=1024, num_leaves = 63, n_estimators=100, max_depth=7, learning_rate = 0.1)
    sf.SetLogFile('/home/share/liangxiao/second_stage/feature_selection/recordml_2.log')
    sf.run(validation)

if __name__ == "__main__":
    main()
