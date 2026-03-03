import numpy as np
from LinReg import LinReg 
import pandas as pd


df = pd.read_csv('U.S._Chronic_Disease_Indicators.csv')
df_X = df[:5]


hi = LinReg(df_X,4)
arr = np.array(['hi', 'hello', 'heyyy', 'whatsup', 'yp'])

print((hi.return_pred(arr)))