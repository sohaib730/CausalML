import random
import pandas as pd
import numpy as np
from scipy.stats import norm,bernoulli,multinomial
from numpy.random import default_rng
rng1 = default_rng()
rng2 = default_rng()
class DataGen:
    def __init__(self,TE,rau,kz,policies,v_dim):
        self.Data_all = pd.DataFrame()
        for i in range(policies):
            AR = random.uniform(.35,.47)
            SR = random.uniform(.49,.5)
            #print (AR,SR)
            data = self.gernerator_procedure(TE,rau,kz,AR,SR,v_dim)
            """TE0 = np.mean(list(data[data['Treatment']==1]['Label_t0']))
            TE1 = np.mean(list(data[data['Treatment']==1]['Label']))
            print ("Decision Policy ",AR)
            print ("Per Policy Treatment Effect ",TE0-TE1)
            print (TE0,TE1)"""
            # 0 -> IH           1->SS           2->OH     #Label is observed outcome
            data['State'] = i+1
            data ['DP'] = AR
            self.Data_all = self.Data_all.append(data)
        #print ('Synthetic Data Shape', self.Data_all.shape)
        #print(self.Data_all.head())

    def gernerator_procedure(self,TE,rau,kz,ar,sr,v_dim=20,n=500):
        #rau = 0.7
        #TE = 0.2
        kv = 25
        mu,sigma = 0.0,1.0
        data_v = []
        data_z = []
        data_w = []


        data_v = norm.rvs(mu,sigma,size=(n,v_dim))
        data_v = np.array(data_v)
        #data_w = norm.rvs(0,1,size=(n,10))
        #data_z = norm.rvs(loc=rau * data_v,scale=1 - rau**2,size=(n,v_dim))
        for i in range(n):
            data_normal = norm.rvs(loc=rau * data_v[i],scale=1 - rau**2,size=v_dim)
            data_z.append(list(data_normal))

        data_v = np.array(data_v)
        data_z = np.array(data_z)
        data_w = np.array(data_w)

        data = np.append(data_v,data_z,axis=1)
        #eps =norm.rvs(size = n, loc = 0 , scale =  0.3 ,random_state = rng2)
        temp_Sum_t0 =  (kv/(kv+rau*kz))*(np.sum(data_v[:,:kv],axis=1) + np.sum(data_z[:,:kz],axis=1))
        #eps =norm.rvs(size = n, loc = 0 , scale =  0.3 ,random_state = rng2)
        temp_Sum_t1 = (kv/(kv+rau*kz))*((np.sum(data_v[:,:kv],axis=1) + np.sum(data_z[:,:kz],axis=1)) - TE*(np.sum(data_v[:,:kv],axis=1) + np.square(np.sum(data_z[:,:kz],axis=1))))
        """eps =norm.rvs(size = n, loc = 0 , scale =  (1/(2*n)) * np.linalg.norm(temp_Sum_t0,ord =2)**2 ,random_state = 10)"""
        eps =norm.rvs(size = n, loc = 0 , scale =  (1/(2*n)) * np.linalg.norm(temp_Sum_t0,ord =2)**2)
        mu_t0 = temp_Sum_t0 + eps
        #eps =norm.rvs(size = n, loc = 0 , scale =  (1/(6*n)) * np.linalg.norm(temp_Sum_t1,ord =2)**2 ,random_state = rng1)
        mu_t1 =temp_Sum_t1 + eps
        temp_true = (kv/(kv+rau*kz)) *(np.sum(data_v[:,:kv],axis=1) + rau*np.sum(data_v[:,:kz],axis=1))
        #mu_true = list(map(lambda x: 1/(1+np.exp(-x)),temp_true ))
        #Output
        #mu_t0 = list(map(lambda x: 1/(1+np.exp(-x)),mu_t0 ))
        #mu_t1 = list(map(lambda x: 1/(1+np.exp(-x)),mu_t1 ))

        Y_t0 = mu_t0#np.array([1 if y >= 0.5 else 0 for y in list(mu_t0)])
        Y_t1 = mu_t1#np.array([1 if y >= 0.5 else 0 for y in list(mu_t1)])

        ### Treatment
        eps =norm.rvs(size = n, loc = 0 , scale =  0.01 )
        temp_Sum =  (1/(kv+kz)**0.5)*(np.sum(data_v[:,:kv],axis=1) + np.sum(data_z[:,:kz],axis=1)) #+ eps

        #eps =norm.rvs(size = n, loc = 0 , scale =  (1/(2*n)) * np.linalg.norm(temp_Sum,ord =2)**2 ,random_state = 10)
        pi = list(map(lambda x: 1/(1+np.exp(-x)),temp_Sum))

        #n = np.count_nonzero( [x for x in pi if x > 0.5] )
        #print ("values under threshold",n)
        #temp_top = sorted(pi)
        """DP_cutoff_index = int(len(temp_top) * ar)    ###r determine likelihood of proposed treatment
        Standard_cutoff_index = int(len(temp_top) * sr)

        DP_cuttoff = temp_top[DP_cutoff_index]
        Standard_cutoff = temp_top[Standard_cutoff_index]"""
        #print (ar,sr)
        def delta(x):
            a = ar-0.01
            b = sr+0.01
            peak = 1.0
            slope = peak / 0.01
            #print (peak)
            if x > a and x < ar:
                return (x-a)*slope
            elif x>=ar and x<= sr:
                return peak
            elif x > sr and x< b:
                return (-1*(x-sr)*slope) + peak
            else:
                return 0.0000001


        def treatment_assignment(x):
            d1 = bernoulli.rvs(min((x/ar)*0.5,0.99),random_state = rng1)
            """if x < 0.5:
                x= max(x-0.25,0.01)
            else:
                x = min(x+0.25,0.99)"""
            d2 = bernoulli.rvs(x,random_state = rng2)
            if d1== 0:
                return d1
            else:# and x < sr:
                return 1+d2



            #random.uniform(0,1)

        A = list(map(treatment_assignment,pi))

        df = pd.DataFrame(data)
        df['Label_t0'] = Y_t0 #list(map(lambda x: max(x,0),Y))
        df['Label_t1'] = Y_t1

        df['Label'] = list(map(lambda x,y0,y1: x*y1 + (1-x)*y0,A,Y_t0,Y_t1))
        df['Treatment'] = A
        #print (df['Treatment'].value_counts())
        #df['v_true'] = v
        df['mu'] = temp_true
        #print (df.head(10))
        return df
