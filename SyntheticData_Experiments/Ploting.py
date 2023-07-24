import seaborn as sns

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.interpolate import interp1d

Result = pd.read_csv("CorrEffect_algo.csv",index_col = False)
Result = Result.drop(Result.columns[0], axis=1)
#x = list(Result['Treatment_Effect'])
#x_interp = np.linspace(x[0], x[-1], 10000)



sns.set_theme()
sns.set_style("ticks")
sns.set_context("paper")
#sns.color_palette("tab10")
#sns.despine()
#sns.set_theme()
"""fig,ax = plt.subplots()
for i,m in enumerate(Method):
    y_values = list(Result[m])
    print (len(x),len(y_values))
    y_interp = interp1d(x,y_values,kind = 'cubic')(x_interp)
    sns.lineplot(x_interp,y_interp,x="Treatment Effect",label = m,ax=ax,markers=True)"""
"""df_melted = Result.melt("Treatment_Effect",var_name="Treatment_type",value_name="Mean Square Error")
#print (df_melted)
ax = sns.lineplot(data=df_melted,x='Treatment_Effect',y='Mean Square Error',hue = 'Treatment_type',palette = 'hls')
#ax.legend(frameon = False, loc = 'upper left',bbox_to_anchor=(1.05,1))
ax.lines[0].set_linestyle("--")"""


Result = Result.set_index(['Correlation'])
print (Result.head())
ax = sns.scatterplot(data=Result,markers = True,palette = 'tab10',s=40)
ax.set(xlabel=r'$\rho$', ylabel='MSE')
#plt.ylim(0,0.7)
#plt.xlim(-0.75,1.00)
#plt.xticks(range(-0.5, 5))
plt.xticks([-0.6,-.5,-.25,0,.25,0.5,0.75, 1])
#sns.despine()
plt.grid()
plt.tight_layout()
plt.show()
