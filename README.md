This is the codebase to reproduce results presented in the paper "Counterfactual Prediction Under Selective Confounding".
# Requirements
	   1. Anaconda (includes jupyter, Python 3.6). Modules required must be installed using "conda install MODULENAME":
		       a) pandas
		       b) lightgbm

## RealWorld Experiments:
	- Launch 'jupyter notebook' and open 'RWData_Experiments/demo_RW_DataSet.ipynb'
		 

## Synthetic Data Experiments:
		a) Run code 'python SyntheticData_Experiments/main_CATE.py' to generate data for Figure 2 in paper
		b) Run code 'python SyntheticData_Experiments/main_correlation.py' to generate data for Figure 3 in paper
		c) Run code 'python SyntheticData_Experiments/main_kz.py' to generate data for Figure 4 in paper
		d) Run code 'python plotting' to get the plots
 
