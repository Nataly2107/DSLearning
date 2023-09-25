
import pandas as pd
import pickle
import warnings
warnings.filterwarnings('ignore')

import yfinance as yf
from prophet import Prophet

rub=yf.download('USDRUB=X', '2020-01-01')
rub=rub.drop('Volume',axis=1)

df=rub
df['Date']=df.index
dff=pd.DataFrame()

dff['ds']=df['Date']
dff['y']=df['Close']
dff=dff.reset_index()
dff.drop(['Date'],axis=1)

m=Prophet()
m.fit(dff)
pickle.dump(m, open("model.pkl", "wb"))


