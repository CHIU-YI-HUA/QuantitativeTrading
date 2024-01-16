# 成果影片
(https://drive.google.com/file/d/1nrMR9fpevemD3V_btbgca1GCUyuOx3tA/view?usp=drive_link)

# Setup

下載GUI code資料夾。

有使用到 numpy 、 tensorflow(OS need to support windows long path) 、 yfinance 、 pandas 、 matplotlib.pyplot 、 numpyencoder 、 seaborn 、 pandas_datareader  等等模組，如本地沒有安裝須事先安裝。

程式使用方法: 執行main.py

# API usage

### project_py:
    
先import py檔
    
        import project

然後建立Agent物件，放入data。
    
        agent = project.Agent(
            state_size = window_size,
            window_size = window_size,
            trend = train_close, #training data
            buy_trend = None,
            skip = skip,
            batch_size = batch_size,
            train_MACD = train_MACD, #training data represented in MACD
            buy_MACD = None,
            MACD_enable = 0)    #  change MACD_enable to 1 if you want to use MACD

我們自己的一些參數

        window_size = 30
        skip = 1
        batch_size = 32
        
訓練和保存模型
            
        agent.train(iterations=200, checkpoint=10, initial_money=10000)
        agent.save_model('save_model_directory/model.ckpt')
建立新Agent物件做測試
    
        agent = project.Agent(
            state_size = window_size,
            window_size = window_size,
            trend = None,
            buy_trend = buy_close,  #test data
            skip = skip,
            batch_size = batch_size,
            train_MACD = None,
            buy_MACD = buy_MACD, #test data represented in MACD
            MACD_enable = test_MACD_enable_var.get() )   #  change MACD_enable to 1 if you want to use MACD

load模型和測試
    
        agent.load_model(openDirectory()+'/model.ckpt')
        states_buy, states_sell, total_gains, invest = agent.buy(initial_money=100000)
        
產生結果圖片
    
        fig = plt.figure(figsize=(15, 5))
        plt.plot(buy_close, color='r', lw=2.)
        plt.plot(buy_close, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
        plt.plot(buy_close, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
        plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
        plt.legend()
        # plt.savefig(''+name+'.png')  # Save the plot as an image
        plt.show()
        
        
        
    
    
### CNN:(訊號燈)

在GUI資料夾中，main.py 和 project.py 以外的其他檔案都是 for CNN 預測

先import py檔
    
        import train as TR
        
   一些固定的參數
                                

        FSize           = 5     
        PSize           = 2
        PStride         = 2
        NumAction       = 3



        # hyper parameters described in the paper 
        #################################################################################
        maxiter         = 5000000       # maxmimum iteration number         
        learning_rate   = 0.00001       # learning rate
        epsilon_min     = 0.1           # minimum epsilon

        W               = 32            # input matrix size
        M               = 1000          # memory buffer capacity
        B               = 10            # parameter theta  update interval               
        C               = 1000          # parameter theta^* update interval ( TargetQ )
        Gamma           = 0.99          # discount factor
        P               = 0             # transaction panalty while training.  0.05 (%) for training, 0 for testing
        Beta            = 32            # batch size
    
先 preprocess data，記得把想預測的股票代號填上。(例：^TWII)
    
        def process():
            data = yf.download("股票代號",period='3mo')

            df = pd.DataFrame(data, columns= ['Volume','Close','Date'])
            df[['Volume','Close']] = df[['Volume','Close']].astype(float)
            df["Date"] = pd.to_datetime(df["Date"])
            data = df.to_numpy()[-32:,:]

            newdata = np.zeros( (32, 3) )
            for i in range(len(data)):
              newdata[i][0] = data[i][0]
              newdata[i][1] = data[i][1]
              newdata[i][2] = i #time
            data = newdata
            #print(data.dtype)

            h = np.shape(data)[0]-31

            conv_data = np.zeros((32,32,h))

            for i in range(h):

                vol_min = data[i:i+32, 0].min()
                vol_max = data[i:i+32, 0].max()
                cost_min = data[i:i+32, 1].min()
                cost_max = data[i:i+32, 1].max()

                temp_data = copy.deepcopy(data[i:i+32, :])
                temp_data[:,0] = (temp_data[:,0] - vol_min)/(vol_max - vol_min) * 14
                temp_data[:,1] = (temp_data[:,1] - cost_min)/(cost_max - cost_min) * 14

                temp_data = np.rint(temp_data)
                temp_data[:,0] = (temp_data[:,0] - 14) * -1
                temp_data[:,1] = ((temp_data[:,1] - 14) * -1) + 17

                for j in range(32):
                    conv_data[int(temp_data[j,0]),j,i] = 1
                    conv_data[int(temp_data[j,1]),j,i] = 1

            conv_data = conv_data.astype(np.int32)
            return conv_data[:,:,i]

restore 已有的模型
    
        Model = TR.trainModel( 1.0, epsilon_min, maxiter, Beta, B , C, learning_rate, P  )
        sess,saver, state, isTrain, rho_eta = Model.TestModel_ConstructGraph( W,W,FSize,PSize,PStride,NumAction )
        saver.restore( sess, 'DeepQ' )

得到預測

        curS = process()
        QAValues = sess.run( rho_eta, feed_dict={ state: curS.reshape(1,W,W), isTrain:False } )
        curA = np.round( QAValues[1].reshape( ( NumAction) ) )
curA為長度3的array，curA[0] == 1 代表建議做多，curA[1] == 1 代表暫不交易，curA[2] == 1 代表建議做空。



# Training.ipynb (在GUI code裡是project .py)
**###Run the Training.ipynb in colab###**

**Import package:**
```
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import deque
from pandas_datareader import data as pdr
import  yfinance as yf
import math
import json
from sklearn.preprocessing import StandardScaler
!pip install yfinance --upgrade --no-cache-dir
```
**Get YahooFinance Stock Data:**

* You can choose the stock you want to train and simulation by setting **data_name**
* **df_train** is the data to train the model
* **df_test** is the data to simulation
```
yf.pdr_override()
data_source = "yfinance"
data_name = "^TWII"
if data_source == "yfinance":
    df_train = pdr.get_data_yahoo(data_name, start="2019-01-01", end="2022-12-31").reset_index()
    df_test = pdr.get_data_yahoo(data_name, start="2023-01-01").reset_index()
```

**Training the model:**

* You can set **MACD_enable** to 1 if you want to use MACD pointer to train the model, or set **MACD_enable** = 0 to only use stock price to train the model
* During training you can see some current model performance, the performance is counted based on the total money earn using the current model
![performance](https://hackmd.io/_uploads/Hku0Y-VtT.png)
* After training you can get the loss rate picture like below
![loss](https://hackmd.io/_uploads/ryvJcbVYa.png)

```
### Initialize
train_close = df_train.Close.values.tolist()
buy_close = df_test.Close.values.tolist()
train_MACD = to_MACD(list(train_close))
buy_MACD = to_MACD(list(buy_close))

initial_money = 1000000
window_size = 30
skip = 1
batch_size = 32

agent = Agent(
    state_size = window_size,
    window_size = window_size,
    trend = train_close,
    buy_trend = buy_close,
    skip = skip,
    batch_size = batch_size,
    train_MACD = train_MACD,
    buy_MACD = buy_MACD,
    MACD_enable = 1    #  change MACD_enable to 1 if you want to use MACD
)

### Train
agent.train(iterations=200, checkpoint=10, initial_money=initial_money)
```
**Simulation and Plot the result:**

* During the Simulation you can see current buying or selling actions
![strategy](https://hackmd.io/_uploads/Hke_cWNKT.png)
* After the Simulation you can see the result of this Simualtion![plot](https://hackmd.io/_uploads/S12u5-Etp.png)

```
### Simulation
states_buy, states_sell, total_gains, invest = agent.buy(initial_money=initial_money)

### Plot
fig = plt.figure(figsize=(15, 5))
plt.plot(buy_close, color='r', lw=2.)
plt.plot(buy_close, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
plt.plot(buy_close, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
plt.legend()
plt.savefig('Simulation.png')  # Save the plot as an image
plt.show()
```
