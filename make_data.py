from statistics import mode
import numpy as np
import pandas as pd
from numpy import newaxis
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential
from datetime import datetime as dt
from datetime import timedelta

def load_data(data_csv, date):
    data = pd.read_csv(data_csv,index_col=date, parse_dates=True, thousands=',', encoding="shift-jis")
    datanull = pd.isnull(data)
    data = data.dropna(how="all")
    #data = data.iloc[::-1]
    data_np = data.to_numpy()
    return data, data_np
# data, data_np = load_data("日経平均00-22日足.csv")

#標準化
def standardization(data, ddof=1):
    mean_data = data.mean()
    std_data = data.std(ddof=ddof)
    standard = (data - mean_data)/(std_data)
    return standard

#対数階差変換
def differ(data):
    y = np.log(data)
    y = data.diff()
    return y

#前日比
def up_or_down(data_close):
    ans = []
    
    for i in range(1,len(data_close)):
        if data_close[i]>data_close[i-1]:
            ans = np.append(ans, 1)
        else: 
            ans = np.append(ans, 0)
    ans = np.append(ans, 0)
    return ans
#data["騰落"] = up_or_down(data["終値"])
def up_or_down_2per(data_close):
    ans = []
    
    for i in range(1,len(data_close)):
        if data_close[i-1]<=data_close[i]<=(data_close[i-1]*1.01):
            ans = np.append(ans, [0,1,0])
        elif (data_close[i-1]*1.01)<data_close[i]:
            ans = np.append(ans, [0,0,1])
        else: 
            ans = np.append(ans, [1,0,0])
    ans = np.append(ans, [1,0,0])
    ans = np.reshape(ans,(len(data_close), 3))
    return ans

def up_or_down_or_stay(data_close):
    ans = []
    for i in range(1,len(data_close)):
        if data_close[i-1]==data_close[i]:
            ans = np.append(ans, [0,1,0])
        elif data_close[i-1]<data_close[i]:
            ans = np.append(ans, [0,0,1])
        else: 
            ans = np.append(ans, [1,0,0])
    ans = np.append(ans, [1,0,0])
    ans = np.reshape(ans,(len(data_close), 3))
    
    return ans

#テクニカル指標作成
#移動平均線
def average_line(data,n):
    ave = data.rolling(n).mean()
    for i in range(n-1):
        ave[i] = data.rolling(i).mean()[i]
    return ave


#クロス移動平均線
def average_line_difference(data_short_line, data_long_line):
    defference = np.array([])
    for i in range(len(data_long_line)):
        d = data_short_line[i] - data_long_line[i]
        defference = np.append(defference, d)
    return defference
#data["5日平均-25日平均"] = average_line_difference(data["5日平均"], data["25日平均"])
#data["25日平均-75日平均"] = average_line_difference(data["25日平均"], data["75日平均"])


#ゴールデンクロス
def cross(data_differ):
    asign = np.sign(data_differ)
    cross = []
    for i in range(len(asign)-1):
        if asign[i]==asign[i+1]:
            cross = np.append(cross, 0)
        elif asign[i]<asign[i+1]:
            cross = np.append(cross, 1)
        else:
            cross = np.append(cross, -1)
    cross = np.append(cross, 0)
    return cross
#data['5-25クロス'] = cross(data["5日平均-25日平均"])
#data['25-75クロス'] = cross(data["25日平均-75日平均"])


#エンベロープ
def upper_envelope(data_close, data_average_line, p):
    env = []
    for i in range(len(data_close)):
        e = data_average_line[i] * (100+p) / 100 - data_close[i]
        env = np.append(env, e)
    return env
def lower_envelope(data_close, data_average_line, p):
    env = []
    for i in range(len(data_close)):
        e = data_average_line[i] * (100-p) / 100 - data_close[i]
        env = np.append(env, e)
    return env
#data["アッパーエンベロープー終値"] = upper_envelope(3, 5, 1, data_np)
#data["ロワーエンベロープー終値"] = lower_envelope(3, 5, 1, data_np)


#指数平滑移動平均線
def ema(closeList=[], term=5):
    return list(pd.Series(closeList).ewm(span=term).mean())
#data["指数平滑移動平均線_26"] = ema(data["終値"], 26)
#data["指数平滑移動平均線_9"] = ema(data["終値"], 9)
#data["MACD"] =  [x - y for (x, y) in zip(ema(data["終値"], 9), ema(data["終値"], 26))]
#data["シグナルMACD"] = average_line(data["MACD"],9)
#data["MACDクロス"] = cross(average_line_difference(data["MACD"], data["シグナルMACD"]))


#ボリンジャーバンド
def bband(data_close):
    bband = pd.DataFrame()
    bband['終値'] = data_close
    bband['mean'] = data_close.rolling(window=20).mean()
    bband['std'] = data_close.rolling(window=20).std()
    bband['upper'] = bband['mean'] + (bband['std'] * 2)
    bband['lower'] = bband['mean'] - (bband['std'] * 2)
    return bband['upper'], bband['lower']
#upper_bband, lower_bband = bband(data)
#data["アッパーボリンジャーバンド"] = upper_bband
#data["ロワーボリンジャーバンド"] = lower_bband
#data["ボリンジャーバンド差"] =  average_line_difference(upper_bband,lower_bband)

def bband_border(upper_bband,lower_bband, high_value, low_value):
    border = []
    for i in range(len(upper_bband)):
        if high_value[i]>upper_bband[i]: border = np.append(border,1)
        elif low_value[i]<lower_bband[i]: border = np.append(border, -1)
        else: border = np.append(border, 0)
    return border
#data["ボリンジャーバンドボーダー"] = bband_border(data["アッパーボリンジャーバンド"], data["ロワーボリンジャーバンド"], data["高値"], data["安値"])


#RSI
def rsi(data_close,num):
    data_diff = data_close.diff()
    data_diff[0]=0
    data_up = []
    data_down = []
    for i in range(len(data_diff)):
        if data_diff[i]>=0:
            data_up.append( data_diff[i])
            data_down.append( 0)
        else:
            data_up.append(0)
            data_down.append(data_diff[i])
    data_up_sum=[]
    data_down_sum=[]
    for i in range(num):
        data_up_sum.append(abs(sum(data_up[:i]))+0.001)
        data_down_sum.append(abs(sum(data_down[:i]))+0.001)
    for i in range(num,len(data_diff)):
        data_up_sum.append(abs(sum(data_up[i-num:i])))
        data_down_sum.append(abs(sum(data_down[i-num:i])))
    rsi = []
    for i in range(len(data_up_sum)):
        rsi.append(0.5-(data_up_sum[i]/(data_up_sum[i]+data_down_sum[i])))
    return rsi

def data_from_year(data_index):
    data_indexx=[]
    data_indexxx=[]
    for i in range(len(data_index)):
        data_indexx.append(str(data_index[i]).replace(' 00:00:00','').split('-'))
        data_indexxx.append(dt(year=int(data_indexx[i][0]),month=int(data_indexx[i][1]),day=int(data_indexx[i][2])))
    data_indexxxx=[]
    data_days=[]
    data_weeks=[]
    for i in range(len(data_indexxx)):
        data_indexxxx.append(data_indexxx[i]-dt(year=data_indexxx[i].year,month=1,day=1))
        data_days.append(data_indexxxx[i].days)
        data_weeks.append(data_days[i]//7)
    return data_days, data_weeks
    