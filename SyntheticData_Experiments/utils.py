import decimal
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.feature_selection import mutual_info_classif
import pandas as pd

#import xgboost as xgb

def float_range(start, stop, step):
    temp = []
    while start < stop:
        temp.append(float(start))
        start = decimal.Decimal(start) + decimal.Decimal(step)
    return temp

def Auc_Score(algo,dtest,y_test):
    #dtest_SS = xgb.DMatrix(X_SS,label = y_SS)
    auc_plot = []
    th_range = float_range(-1,1,'0.1')
    for th in th_range:
        y_det = algo.predict(dtest) > th

        #print (algo.predict_proba(X_test)[:,1][:10],y_det[:10])
        #print (f"{method} test Accuracy",accuracy_score(y_det, y_test))
        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_det).ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(tn+fp)
        auc_plot.append((tpr,fpr))
    tpr_temp = [a_tuple[0] for a_tuple in auc_plot]
    fpr_temp = [a_tuple[1] for a_tuple in auc_plot]
    auc_score = "{:.2f}".format(metrics.auc(fpr_temp, tpr_temp))
    return auc_score
def Auc_Score_Classifier(algo,X_test,y_test):
    #dtest = xgb.DMatrix(X_test,label = y_test)
    auc_plot = []
    th_range = float_range(-1,1,'0.1')
    for th in th_range:
        y_det = algo.predict_proba(X_test)[:,1] > th
        #print (y_det.shape)
        #print (y_test.shape)
        #print (y_det[:10])
        #print (algo.predict_proba(X_test)[:,1][:10],y_det[:10])
        #print (f"{method} test Accuracy",accuracy_score(y_det, y_test))

        tn, fp, fn, tp = metrics.confusion_matrix(y_test, y_det).ravel()
        tpr = tp/(tp+fn)
        fpr = fp/(tn+fp)
        auc_plot.append((tpr,fpr))
    tpr_temp = [a_tuple[0] for a_tuple in auc_plot]
    fpr_temp = [a_tuple[1] for a_tuple in auc_plot]
    auc_score = "{:.2f}".format(metrics.auc(fpr_temp, tpr_temp))
    return auc_score
def mse(algo,X_test,y_test):
    y_det = algo.predict(X_test)
    result = list(map(lambda x,y: (x-y)**2,y_det,y_test))
    result = (1/y_det.shape[0]) * sum(result)
    return result
def mse_ATE(y_det,y_test):
    result = list(map(lambda x,y: (x-y)**2,y_det,y_test))
    result = (1/y_det.shape[0]) * sum(result)
    return result
def report(y_test,y_pred):
    target_names = ['Safe','MalTreatment_12']
    print(classification_report(y_test, y_pred, target_names=target_names))
def make_mi_scores(X, y, discrete_features):
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    print (mi_scores[:10])  # show a few features with their MI scores
