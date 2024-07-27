import pandas as pd
import datetime
import FinanceDataReader as fdr
import matplotlib.pyplot as plt
import requests
import schedule
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pykrx import stock
from pykrx import bond
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import time






#1)어제 등락률 상위 10개
yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")

ohlcv = stock.get_market_ohlcv(yesterday, market="ALL")
top_10=ohlcv.sort_values(by='등락률',ascending=False).head(10)
top_10_tickers = top_10.index.tolist()
top_10_tickers

#2)코스피, 코스닥
today = datetime.today().strftime("%Y-%m-%d")
months_ago= (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")

kospi = fdr.DataReader('KS11',months_ago)
kospi['Close'].plot()


kosdaq = fdr.DataReader('KQ11',months_ago)
kosdaq['Close'].plot()

#3)금일 주가 예측

df = fdr.DataReader('005930', '2017-01-01', today)
df.shape

def MinMaxScaler(data):
    """최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]
dfy
dfx
X = dfx.values.tolist()
y = dfy.values.tolist()

window_size = 10

data_X = []
data_y = []
for i in range(len(y) - window_size):
    _X = X[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]     # 다음 날 종가
    data_X.append(_X)
    data_y.append(_y)
print(_X, "->", _y)

train_size = int(len(data_y) * 0.7)
train_X = np.array(data_X[0 : train_size])
train_y = np.array(data_y[0 : train_size])

test_size = len(data_y) - train_size
test_X = np.array(data_X[train_size : len(data_X)])
test_y = np.array(data_y[train_size : len(data_y)])

print('훈련 데이터의 크기 :', train_X.shape, train_y.shape)
print('테스트 데이터의 크기 :', test_X.shape, test_y.shape)

model = Sequential()
model.add(LSTM(units=20, activation='relu', return_sequences=True, input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(LSTM(units=20, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(train_X, train_y, epochs=70, batch_size=30)
pred_y = model.predict(test_X)

plt.figure()
plt.plot(test_y, color='red', label='real SEC stock price')
plt.plot(pred_y, color='blue', label='predicted SEC stock price')
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.show()
print("내일 SEC 주가 :", df.Close[-1] * pred_y[-1] / dfy.Close[-1], 'KRW')


#4)나라별 환전

usd_krw = fdr.DataReader('USD/KRW', today)
eur_krw = fdr.DataReader('EUR/KRW', today)
jpy_krw = fdr.DataReader('JPY/KRW', today)
cny_krw = fdr.DataReader('CNY/KRW', today)

#5)매주 ai추천
def job() :
    days_ago = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
    days_ago2 = (datetime.today() - timedelta(days=2)).strftime("%Y%m%d")
    days_ago3 = (datetime.today() - timedelta(days=3)).strftime("%Y%m%d")
    
    df_3 = stock.get_market_ohlcv(days_ago3, market="ALL")
    df_2 = stock.get_market_ohlcv(days_ago2, market="ALL")
    df_1 = stock.get_market_ohlcv(days_ago, market="ALL")
    
    df_3 = df_3[['등락률']]
    df_2 = df_2[['등락률']]
    df_1 = df_1[['등락률']]
    
    # 빈 데이터프레임 생성
    df_all = pd.DataFrame()
    
    # 데이터프레임 리스트
    dfs = [df_1, df_2, df_3]
    
    # 빈 데이터프레임에 새로운 컬럼 추가
    for i, df in enumerate(dfs, start=1):
        col_name = f'등락률_{i}일전'
        
        df_all[col_name] = df['등락률']
    
    print(df_all)
    df_all = df_all[(df_all['등락률_1일전'] > 0) | (df_all['등락률_2일전'] > 0) | (df_all['등락률_3일전'] > 0)]
    
    df_all['등락률 합'] = df_all['등락률_1일전'] + df_all['등락률_2일전'] + df_all['등락률_3일전']
    
    top_5_all=df_all.sort_values(by='등락률 합',ascending=False).head(5)
    top_5_all_tickers = top_5_all.index.tolist()
    top_5_all_tickers
    

schedule.every().thursday.at("09:00").do(job)

days_ago = (datetime.today() - timedelta(days=1)).strftime("%Y%m%d")
days_ago2 = (datetime.today() - timedelta(days=2)).strftime("%Y%m%d")
days_ago3 = (datetime.today() - timedelta(days=3)).strftime("%Y%m%d")

df_3 = stock.get_market_ohlcv(days_ago3, market="ALL")
df_2 = stock.get_market_ohlcv(days_ago2, market="ALL")
df_1 = stock.get_market_ohlcv(days_ago, market="ALL")

df_3 = df_3[['등락률']]
df_2 = df_2[['등락률']]
df_1 = df_1[['등락률']]

# 빈 데이터프레임 생성
df_all = pd.DataFrame()

# 데이터프레임 리스트
dfs = [df_1, df_2, df_3]

# 빈 데이터프레임에 새로운 컬럼 추가
for i, df in enumerate(dfs, start=1):
    col_name = f'등락률_{i}일전'
    
    df_all[col_name] = df['등락률']

print(df_all)
df_all = df_all[(df_all['등락률_1일전'] > 0) | (df_all['등락률_2일전'] > 0) | (df_all['등락률_3일전'] > 0)]

df_all['등락률 합'] = df_all['등락률_1일전'] + df_all['등락률_2일전'] + df_all['등락률_3일전']

top_5_all=df_all.sort_values(by='등락률 합',ascending=False).head(5)
top_5_all_tickers = top_5_all.index.tolist()
top_5_all_tickers