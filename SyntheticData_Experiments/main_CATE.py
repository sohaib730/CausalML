from CGen_Data import DataGen
from ATE_Estimator import ATE_Learner
from Double_Robust import DRLearner
from utils import *
import seaborn as sns
sns.set(style="darkgrid")
from sklearn.model_selection import cross_val_predict, KFold, train_test_split
import matplotlib.pyplot as plt

if __name__ == "__main__":


    setting = ['R','N','C']   ##['No calibration',"oracle y0","estimated y0"]
    Result = pd.DataFrame(columns = setting)

    v_dim = 200

    TE = [0,0.25,0.5,0.75,1.0,1.25,1.5,1.75] #,2.1,2.2,2.3,2.4,2.5,2.6,2.7,2.8,2.9,3.0,3.1,3.2,3.3,3.4,3.5,3.6,3.7,3.8,3.9,4.0]
    for te in TE:
        setting_score = {}
        for _ in range(2):
            policies = 10
            SYN_data = DataGen(te,0.25,20,policies,v_dim)
            train, test = train_test_split(SYN_data.Data_all, test_size=0.20, random_state=3, shuffle=True)
            test_copy = test.copy()
            y_test = test.pop('mu')       #### testing it against desired label
            T_test = test.pop('Treatment')
            print ("Test Treatment Counts",T_test.value_counts())
            #test.pop('mu')
            X_test = test.iloc[:,:v_dim]
            ATE_est = ATE_Learner(train,policies,v_dim)
            for sett in setting:
                #if sett == 'C':

                DR = DRLearner(train.copy(),test_copy,sett,v_dim,ATE_est.calib_param,ATE_est.intercept)

                MSE_score = mse(DR.DR_algo,X_test,y_test)
                #avg_score += MSE_score
                setting_score[sett] = setting_score.get(sett,0) + MSE_score

        for k in setting_score:
            setting_score[k] /= 2.0
        #setting_score /= 2.0
        setting_score['Treatment_Effect'] = te
        Result = Result.append(setting_score,ignore_index = True)
            #result_MSE_Pred.append(avg_score/2.0)
    print ("Prediction error with varying Treatment effect",Result)
        #Result[sett] = result_MSE_Pred

    Result.to_csv("TreatmentEffect_Result.csv")
