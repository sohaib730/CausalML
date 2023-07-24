from sklearn.linear_model import LogisticRegression,Ridge,SGDClassifier


from lightgbm import LGBMRegressor
import lightgbm as lgb
import xgboost as xgb
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
    def __init__(self, train,test, setting, v_dim,ATE_est,eval_algo = None):
        if setting == 'R':
            train.pop('Label')
            train.loc[:,'Label'] = t_y0 #assigning ground truth y0 label, only short stayer will be used
            print ("Oracle access to Label")
        elif setting == 'C':
            # Calibration of Short Stayer T=1 Label
            #print (calib_setting)
            #train[train.Treatment == 1]['Label'] = -1# +=  np.array(train[train.Treatment == 1].iloc[:,:v_dim]).dot(calib_setting)
            train.loc[train.Treatment == 2, 'Label'] += ATE_est.intercept
            train.loc[train.Treatment == 2, 'Label'] += np.array(train[train.Treatment == 2].iloc[:,:v_dim]).dot(ATE_est.calib_param)
            #print (train[train.Treatment == 1].loc['Label'])
            print ("Calibration Performed")
        else:
            print ("No callibration done")



        """cv = KFold(n_splits=2, shuffle=True, random_state=30)
        split_indices = [index for _, index in cv.split(train)]
        #CV_model = []
        #for i in range(2):
        db0 = train.iloc[split_indices[0]]
        db1 = train.iloc[split_indices[1]]"""
        #print ("######Training Double Robust####### ")
        SS_train = train[train['Treatment']==2]  #Short Stayer
        #print ("Short Stayer Train Shape",SS_train.shape)

        self.feat_size = 66 #int(v_dim*2)

        X,y,T = split_feature(train[train['Treatment']==2])
        self.Direct_Method(X.iloc[:,:self.feat_size],y,test[test.Treatment == 2])   # Finding mu_1(x)


        X,y,T = split_feature(train[train.Treatment != 0])
        self.Propensity_Score(X.iloc[:,:self.feat_size],T[:]%2)  ## pi  ##SS Treatment should be 1

        X,y,T = split_feature(train)
        self.train(X.iloc[:,:self.feat_size],y,T,v_dim)

        X,y,T = split_feature(train)
        self.train_RA(X.iloc[:,:self.feat_size],y,T,v_dim)

        X,y,T = split_feature(train[train.Treatment == 0])
        self.Stand_Prac(X,y,v_dim)

        X,y,T = split_feature(train[train.Treatment == 2])
        if not eval_algo:
            return
        elif eval_algo == 'DR':
            y_det = self.DR_algo.predict(X.iloc[:,:v_dim])
        elif eval_algo == 'RA':
            y_det = self.RA_algo.predict(X.iloc[:,:v_dim])
        else:
            y_det = self.SP_algo.predict(X.iloc[:,:v_dim])
        mse_y = list(map(lambda x,y: (x-y)**2,y,y_det))
        self.eval_step1(X,mse_y)



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
        y_test = test['Label']

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            class_weights = list(class_weight.compute_class_weight('balanced',
                                             np.unique(y_train),
                                             y_train))
            #print (class_weights)
            c_weight = {0:class_weights[0],1:class_weights[1]}
        #self.DM_algo = lgb.train(params = param,train_set = data_train, valid_sets = data_val,early_stopping_rounds=10,verbose_eval = False)
        self.DM_algo = LinearRegression().fit(X_train,y_train)
        #print (self.DM_algo.feature_importance(importance_type='gain')[:10])
        auc = mse(self.DM_algo,X_test,y_test)
        #print (f"AUC Score for Direct Method over SS data: {auc}")

    def Stand_Prac(self,X,y,v_size):
        self.SP_algo = LinearRegression().fit(X.iloc[:,:v_size],y)
        #self.DA_algo = LGBMRegressor().fit(X.iloc[:,:80],y)

    def Propensity_Score(self,X,T):
        #print (T.iloc[:10])
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
        #self.PS_algo = RandomForestClassifier(max_depth = 9,random_state = 43).fit(X_train,y_train)
        self.PS_algo = LinearRegression().fit(X_train,y_train)
        auc = mse(self.PS_algo,X_test,y_test)
        #print (f"AUC Score for Propensity_Score over OH Samples: {auc}")

    def train(self,X,y,T,v_size):
        print ("\n Train DR using all treatment Samples")
        #pi_t = self.PS_algo.predict_proba(X)[:,1]   ###Pred probability for SS
        pi_t = self.PS_algo.predict(X)
        mu_t = self.DM_algo.predict(X)

        sudo_outcome = mu_t
        count = 0
        for i in range(X.shape[0]):
            if T.iloc[i] == 0:   ##Treatment for In-home
                sudo_outcome[i] = y.iloc[i]
            elif T.iloc[i] == 2:    ## Treatment for short stayer
                sudo_outcome[i] = (sudo_outcome[i] +  (1/min(1.0,pi_t[i]+0.1)) * (y.iloc[i] - mu_t[i]))
        self.DR_algo = LinearRegression().fit(X.iloc[:,:v_size],sudo_outcome)

    def train_RA(self,X,y,T,v_size):
        print ("\n Train DM using all treatment train Samples")
        mu_t = self.DM_algo.predict(X)
        sudo_outcome = mu_t
        count = 0
        for i in range(X.shape[0]):
            if T.iloc[i] == 0:
                sudo_outcome[i] = y.iloc[i]
        self.RA_algo = LinearRegression().fit(X.iloc[:,:v_size],sudo_outcome)

    def eval_step1(self,train_x,train_y):
        print ("eta train shape",train_x.shape)
        X_train = train_x
        y_train = train_y
        self.eta_algo = LinearRegression().fit(X_train,y_train)
