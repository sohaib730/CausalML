from sklearn.linear_model import LogisticRegression,Ridge,SGDClassifier,LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor

from lightgbm import LGBMRegressor
import lightgbm as lgb
#import xgboost as xgb
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
import numpy as np

from utils import *
import warnings


def split_feature(db):
    X = db.copy()
    y = X.pop('Label')
    #print(y.iloc[:10])
    T = X.pop('Treatment')
    return X,y,T
class DRLearner:
    ## DM is Regressive model
    ## Propensity is Classifier
    ### Doubly robust model is again regression .
    def __init__(self, train,test, setting, v_dim,calib_setting,intercept):
        train.pop('mu')
        t_y0 = list(train.pop('Label_t0'))
        t_y1 = train.pop('Label_t1')
        #if np.sum(calib_setting) < 1.0:
            #calib_setting *= 2
        print (calib_setting[:10])
        train_OH = train[train['Treatment']!=0]
        if setting == 'R':
            train.pop('Label')
            train.loc[:,'Label'] = t_y0 #assigning ground truth y0 label, only short stayer will be used
            print ("Oracle access to Label")
        elif setting == 'C':
            # Calibration of Short Stayer T=1 Label
            #print (calib_setting)
            #train[train.Treatment == 1]['Label'] = -1# +=  np.array(train[train.Treatment == 1].iloc[:,:v_dim]).dot(calib_setting)
            train.loc[train.Treatment == 1, 'Label'] += intercept
            train.loc[train.Treatment == 1, 'Label'] += np.array(train[train.Treatment == 1].iloc[:,:v_dim]).dot(calib_setting)
            #print (train[train.Treatment == 1].loc['Label'])
            print ("Calibration Performed")
        else:
            print ("No callibration done")



        cv = KFold(n_splits=2, shuffle=True)
        split_indices = [index for _, index in cv.split(train)]
        #CV_model = []
        #for i in range(2):
        db0 = train.iloc[split_indices[0]]
        db1 = train.iloc[split_indices[1]]
        #print ("######Training Double Robust####### ")
        SS_train = train[train['Treatment']==1]  #Short Stayer
        print ("Short Stayer Train Shape",SS_train.shape)

        self.feat_size = int(v_dim*2)

        X,y,T = split_feature(train[train['Treatment']==1])
        self.Direct_Method(X.iloc[:,:self.feat_size],y,test[test.Treatment == 1])   # Finding mu_1(x)


        X,y,T = split_feature(train[train.Treatment != 0])
        self.Propensity_Score(X.iloc[:,:self.feat_size],T)  ## pi  ##SS Treatment should be 1

        X,y,T = split_feature(train)
        self.train(X.iloc[:,:self.feat_size],y,T,v_dim)

        X,y,T = split_feature(train)
        self.train_RA(X.iloc[:,:self.feat_size],y,T,v_dim)

        X,y,T = split_feature(train[train.Treatment == 0])
        self.Stand_Prac(X,y,v_dim)



    def Direct_Method(self,train_x,train_y,test):
        print ("DM Train Shape",train_x.shape)

        ### This model should be regressive model not classifier
        #X_train, X_test, y_train, y_test  = train_test_split(train_x, train_y, test_size=0.05,random_state=3)
        #print (y_test[:20])
        #print ("Abuse Count Ratio",np.count_nonzero(y_train==1)/y_train.shape[0])
        X_train = train_x
        y_train = train_y
        #print (y_train[:10])
        X_test = test.iloc[:,:self.feat_size]
        y_test = test['Label_t0']

        """with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            class_weights = list(class_weight.compute_class_weight('balanced',
                                             np.unique(y_train),
                                             y_train))
            #print (class_weights)
            c_weight = {0:class_weights[0],1:class_weights[1]}"""
        #self.DM_algo = lgb.train(params = param,train_set = data_train, valid_sets = data_val,early_stopping_rounds=10,verbose_eval = False)
        self.DM_algo = LGBMRegressor().fit(X_train,y_train)
        #print (self.DM_algo.feature_importance(importance_type='gain')[:10])
        #auc = Auc_Score(self.DM_algo,X_test,y_test)
        #print (f"AUC Score for Direct Method over SS data: {auc}")

    def Stand_Prac(self,X,y,v_size):
        self.SP_algo = LGBMRegressor().fit(X.iloc[:,:v_size],y)
        #self.DA_algo = LGBMRegressor().fit(X.iloc[:,:80],y)

    def Propensity_Score(self,X,T):
        #print (T.iloc[:10])
        T= list(map(lambda t: t%2,T))
        #print (T[:10])
        X_train, X_test, y_train, y_test  = train_test_split(X, T, test_size=0.20)

        #print ("\nShort Stayer Ratio",np.count_nonzero(y_train==1)/y_train.shape[0])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            class_weights = list(class_weight.compute_class_weight('balanced',
                                                 np.unique(T),
                                                 T))
            #print (class_weights)
            weight = {0:class_weights[0],1:class_weights[1]}
        #print (weight)
        self.PS_algo = RandomForestClassifier(class_weight = weight).fit(X,T)
        #self.PS_algo = LGBMRegressor().fit(X,T)
        #auc = Auc_Score(self.PS_algo,X_test,y_test)
        #print (f"AUC Score for Propensity_Score over OH Samples: {auc}")

    def train(self,X,y,T,v_size):
        print ("\n Train DR using all treatment Samples")
        pi_t = self.PS_algo.predict_proba(X)[:,1]   ###Pred probability for SS
        #pi_t = self.PS_algo.predict(X)
        mu_t = self.DM_algo.predict(X)

        sudo_outcome = mu_t
        count = 0
        for i in range(X.shape[0]):
            if T.iloc[i] == 0:
                sudo_outcome[i] = y.iloc[i]
            elif T.iloc[i] == 1:    ## Treatment for short stayer
                sudo_outcome[i] = mu_t[i] +  (1/pi_t[i]) * (y.iloc[i] - mu_t[i])
        self.DR_algo = LGBMRegressor().fit(X.iloc[:,:v_size],sudo_outcome)

    def train_RA(self,X,y,T,v_size):
        print ("\n Train DM using all treatment train Samples")
        mu_t = self.DM_algo.predict(X)
        sudo_outcome = mu_t
        count = 0
        for i in range(X.shape[0]):
            if T.iloc[i] == 0:
                sudo_outcome[i] = y.iloc[i]
        self.RA_algo = LGBMRegressor().fit(X.iloc[:,:v_size],sudo_outcome)
