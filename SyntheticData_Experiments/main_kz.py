from CGen_Data import DataGen
from ATE_Estimator import ATE_Learner
from Double_Robust import DRLearner
from utils import *
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":


    algorithm = ['DR','RA','SP']   ##['No calibration',"oracle y0","estimated y0"]
    Result = pd.DataFrame()
    kz = [0,5,10,15,20,25,30,35,40]
    if Result.empty:
        Result['kz'] = kz
    DR_result = []
    RA_result = []
    SP_result = []
    for conf in kz:
        sum_ATE = 0
        MSE_score_DR = 0
        MSE_score_RA = 0
        MSE_score_SP = 0
        for _ in range(200):
            policies = 5
            v_dim = 250
            #SYN_data = DataGen(policies,corr,v_dim)
            SYN_data = DataGen(0.1,0.25,conf,policies,v_dim)
            """TE0 = np.mean(list(SYN_data.Data_all[SYN_data.Data_all['Treatment']==1]['Label_t0']))
            TE1 = np.mean(list(SYN_data.Data_all[SYN_data.Data_all['Treatment']==1]['Label']))
            #print ("Treatment Effect ",TE0-TE1)
            ate = TE0 - TE1
            print ("Treatment Effect",ate)"""
            #SYN_data.Data_all[SYN_data.Data_all['Treatment']==1]['Label'] +=  ate
            """print ('Data Statistics')
            print (SYN_data.Data_all['Treatment'].value_counts())
            print ("IH Labels",SYN_data.Data_all[SYN_data.Data_all['Treatment']==0]['Label_t0'].value_counts())
            print ("SS Labels",SYN_data.Data_all[SYN_data.Data_all['Treatment']==1]['Label_t0'].value_counts())
            print ("SS Labels_t1",SYN_data.Data_all[SYN_data.Data_all['Treatment']==1]['Label_t1'].value_counts())
            print ("OH Labels",SYN_data.Data_all[SYN_data.Data_all['Treatment']==2]['Label_t0'].value_counts())"""
            #print (SYN_data.Data_all.iloc[:10,0:5],SYN_data.Data_all.iloc[:10,20:25])
            train, test = train_test_split(SYN_data.Data_all, test_size=0.20, random_state=3, shuffle=True)
            #test = test[test.Treatment!=0]
            test_copy = test.copy()
            y_test = test.pop('mu')       #### testing it against desired label
            T_test = test.pop('Treatment')
            print ("Test Treatment Counts",T_test.value_counts())
            #test.pop('mu')
            X_test = test.iloc[:,:v_dim]
            ATE_est = ATE_Learner(train,policies,v_dim)

            #result = mse(ATE_est.ate_reg,test[test['Treatment']==1].iloc[:,:80],test[test['Treatment']==1]['Label_t0'])
            #print ("MSE y0 Prediction for short stayer",result)
            #sum_ATE += result
            ## Doubly Robust Estimation
            #print (train.head())
            #db2 = train.iloc[split_indices[(i+2)%3]]
            #print (train.head())
            DR = DRLearner(train.copy(),test_copy,'C',v_dim,ATE_est.calib_param,ATE_est.intercept)
            #DR = DRLearner(train,'C',v_dim)
            #CV_model.append(DR)
            #print ("\n*****************Evaluate DR over test IH,SS,OH Samples***********")
            #test = test[test['Treatment']==2]
            #test.pop('mu')       #### testing it against desired label

            #test.pop('v_true')
            #vhat = [0] * y.shape[0]
            #for i in range(2):
            #print( DR.NA_algo.predict(X.iloc[:10]),y.iloc[:10])
            for algo in algorithm:
                if algo == 'DR':
                    MSE_score_DR += mse(DR.DR_algo,X_test,y_test)
                elif algo == 'RA':
                    MSE_score_RA += mse(DR.RA_algo,X_test,y_test)
                else:
                    MSE_score_SP += mse(DR.SP_algo,X_test,y_test)


            #print (f"MSE E(y - y') DR Prediction: {MSE_score}")
            #auc = Auc_Score(DR.DR_algo,X,y)
            #print (f"AUC for DR Prediction: {auc}")
        #print ("Final Result y0 estimation for Short Stayer",sum_ATE/3.0)
        print ("Final Prediction MSE DR",MSE_score_DR/200.0)
        print ("Final Prediction MSE RA",MSE_score_RA/200.0)
        print ("Final Prediction MSE SP",MSE_score_SP/200.0)
        DR_result.append(MSE_score_DR/200.0)
        RA_result.append(MSE_score_RA/200.0)
        SP_result.append(MSE_score_SP/200.0)

    Result['DR'] = DR_result
    Result['RA'] = RA_result
    Result['SP'] = SP_result
        #Result[algo] = result_MSE_Pred
    Result.to_csv(f"ConfEffect_algo.csv")
