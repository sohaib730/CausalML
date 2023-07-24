import random
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import KNeighborsRegressor,KNeighborsClassifier
from lightgbm import LGBMRegressor
from sklearn.linear_model import LogisticRegression,Ridge,SGDClassifier
from sklearn.svm import OneClassSVM
from utils import *
from sklearn.linear_model import LinearRegression

class ATE_Learner:
    def __init__(self,train,policies,v_dim):
        self.num_state = policies
        self.dim = v_dim
        #pdata = self.create_pdata(train)
        #self.ate_reg = self.train(pdata)
        self.estimate_callibration(train)
    def y0_estimator(self,train):
        R0_reg =  LGBMRegressor().fit(train[train['Treatment']==0].iloc[:,:40],train[train['Treatment']==0]['Label'])
        return R0_reg
    def train(self,pdata):
        reg_y0  = LGBMRegressor().fit(pdata.iloc[:,:self.dim*2],pdata['Label_est_y0'])
        return reg_y0
    def estimate_callibration(self,train):
        # Train Omega_a(X)
        Reg_0_X =  LGBMRegressor().fit(train[train['Treatment']==0].iloc[:,:self.dim],train[train['Treatment']==0]['Label'])
        # predict Label for dual treatment samples
        Y_a_Pred = Reg_0_X.predict(train[train['Treatment']==2].iloc[:,:self.dim])
        Label_diff = Y_a_Pred -  train[train['Treatment']==2]['Label']
        reg = LinearRegression().fit(train[train['Treatment']==2].iloc[:,:self.dim], Label_diff)
        self.calib_param = reg.coef_
        self.intercept = reg.intercept_
        #print("Intercept",self.intercept)

    def create_pdata_ne(self,train):
        train.pop('mu')
        train.pop('Label_t1')
        # Still 4 feature remaining "State,Treatment,Label,Label_t0"
        est_t = 2  ## Treatment for which y0 needs to be learned
        pseudo_train_data = pd.DataFrame()
        for k in range(self.num_state):
            each_state = train[train['State']==k+1].copy()
            if each_state[each_state['Treatment']==1].shape[0] == 0:
                continue
            label = each_state.pop('Label')
            neigh = KNeighborsRegressor(n_neighbors=3)
            neigh.fit(each_state[each_state['Treatment']==0].iloc[:,:self.dim], label[each_state['Treatment']==0])
            SS_y0 = neigh.predict(each_state[each_state['Treatment']==est_t].iloc[:,:self.dim])
            #if SS_y0.shape[1] == 2:
            outlier = each_state[each_state['Treatment']==est_t].copy()
            outlier['Label_est_y0'] = [y for y in SS_y0]
                #outlier['Label_y1'] = label[ea]
            pseudo_train_data = pseudo_train_data.append(outlier)
        print("ATE over pseudo data",mse_ATE(pseudo_train_data['Label_est_y0'],pseudo_train_data['Label_t0']))
        print ("Pseudo data shape",pseudo_train_data.shape)
        return pseudo_train_data

    def create_pdata(self,train):
        train.pop('mu')
        train.pop('Label_t1')
        # Still 4 feature remaining "State,Treatment,Label,Label_t0"
        est_t = 1  ## Treatment for which y0 needs to be learned
        pseudo_train_data = pd.DataFrame()
        for k in range(self.num_state):
            each_state = train[train['State']==k+1].copy()
            label = each_state.pop('Label')
            base_t0 = each_state.pop('Label_t0')
            #each_state.pop('State')
            #print (f"For State {k} Treatment a' No fuss Mean {label[each_state['Treatment']==1].mean()}")
            if each_state[each_state['Treatment']==est_t].shape[0] <=10:
                continue
            clf = IsolationForest(max_features = 20,contamination = 0.01).fit(each_state[each_state['Treatment']!=0].iloc[:,:self.dim*2])
            outliers = pd.DataFrame()
            for j in range(self.num_state):
                if j == k:
                    continue
                other_state = train[train['State']==j+1].copy()
                if other_state[other_state['Treatment']==est_t].shape[0] == 0:    #No samples
                    continue
                t_label = other_state.pop('Label')
                t_label_t0 = other_state.pop('Label_t0')
                prediction = clf.predict(other_state[other_state['Treatment']==est_t].iloc[:,:self.dim*2])
                out_index = [i for i,x in enumerate(prediction) if x == -1]    ###outlie prediction
                if not out_index:
                    continue
                other_state['Label'] = t_label
                other_state['Label_t0'] = t_label_t0
                outliers = outliers.append(other_state[other_state['Treatment']==est_t].iloc[out_index])

            # Predict y_0 labels for each_state outlier
            print ("Outlier Shape",outliers.shape[0])
            if outliers.shape[0] == 0:
                continue
            #print (f"For State {k} Treatment a' Mean {outliers['Label'].mean()}")
            #print (f"For State {k} Treatment a True Mean {outliers['Label_t0'].mean()}")
            t_try = list(label[each_state['Treatment']==0])
            #fraction = outliers.shape[0]/len(t_try)
            t_try = sorted(t_try,reverse=True)
            t_try = t_try[:outliers.shape[0]]
            TE0 = np.mean(t_try)
            TE1 = np.mean(list(outliers['Label']))
            ATE = TE0 - TE1
            print ("ATE_est",TE0 - TE1)
            TE0 = np.mean(list(base_t0[each_state['Treatment']==1]))
            TE1 = np.mean(list(label[each_state['Treatment']==1]))
            ATE = TE0 - TE1
            print ("ATE_true",TE0 - TE1)


            #print ("Outliers Stat",[1 if y >= 0.5 else 0 for y in list(outliers['Label'])].value_counts())
            #print (f"For State {k} Treatment a Measured Mean {np.mean(t_try)}")

            neigh = KNeighborsRegressor(n_neighbors=10)
            neigh.fit(each_state[each_state['Treatment']==0].iloc[:,:self.dim], label[each_state['Treatment']==0])
            #neigh.fit(each_state[each_state['Treatment']==0].iloc[:,:self.dim], label[each_state['Treatment']==0])

            outliers_y_0 = neigh.predict(outliers.iloc[:,:self.dim])
            """if outliers_y_0.shape[1] == 1:
                continue"""

            outliers['Label_est_y0'] = outliers_y_0 #outliers['Label']+ATE # outliers_y_0 #[y for x,y in outliers_y_0]
            #outliers['Label_ate_y0'] = outliers[]
            pseudo_train_data = pseudo_train_data.append(outliers)
            print (f"For State {k} Treatment a Measured KNN Mean {pseudo_train_data['Label_est_y0'].mean()}")
            #print ("Overall predicted Treatment effect",np.average(base_t0) - np.average(outliers['Label']))
            #print ("Conditional predicted Treatment effect",np.average(outliers['Label_ate_y0']) )
            #print ("Pseudo data shape after each run",pseudo_train_data.shape)
        #print("ATE over pseudo data",mse_ATE(pseudo_train_data['Label_ate_y0'],pseudo_train_data['Label_t0'] - pseudo_train_data['Label']))
        #print ("Pseudo data shape",pseudo_train_data.shape)
        return pseudo_train_data

    def caliberate(self,train):
        for k in range(self.num_state):
            mask_k = (train.State == k+1) & (train.Treatment == 1)
            size = train.loc[mask_k].shape[0]
            #train.loc[mask_k,'Label'] = 0
            #each_state.loc['Label'] = 0
            TE1 = np.mean(list(train.loc[mask_k,'Label']))
            TE = 0
            ignore = 1
            for j in range(self.num_state):
                if j == k:
                    continue
                other_state = train.loc[train.State == j+1,:]
                mask_j = (train.State == j+1) & (train.Treatment == 0)
                temp_t0 = list(train.loc[mask_j,'Label'])
                temp_t0 = sorted(temp_t0,reverse=True)
                L_ATE = np.mean(temp_t0[:int(.75*size)]) - TE1
                TE += L_ATE
                """if L_ATE > 0.02 :
                    TE += L_ATE
                else:
                    ignore += 1"""
            if self.num_state - ignore != 0:
                TE /= (self.num_state - ignore)
            else:
                TE = 0
            self.Est_TE = TE
            #self.true_TE = np.mean(list(train.loc[mask_k,'Label_t0'])) - np.mean(list(train.loc[mask_k,'Label']))
            print (f"State {k+1} Estimated Treatment Effect {TE}")
            train.loc[mask_k,'Label'] += TE
            #each_state.loc[each_state.Treatment == 1,'Label'] = 100
        #print (train[train.Treatment == 1]['Label'][:10])
