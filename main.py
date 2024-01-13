import DataPPRL as DRL
import train as TR
import numpy as np
import yfinance as yf
import pandas as pd
import copy
import matplotlib.pyplot as plt
import tkinter as tk  # 使用Tkinter前需要先导入
from tkinter import ttk
import datetime
import project
from tkinter import filedialog
import time
import tkinter.messagebox

##############################################################################
# tensorflow  1.2.0
# Ubuntu 16.04
##############################################################################

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
#################################################################################

# initialize
DRead           = DRL.DataReaderRL()
Model           = TR.trainModel( 1.0, epsilon_min, maxiter, Beta, B , C, learning_rate, P  )

#record_initial_money = 0
try:
    stock_data = pd.read_csv("stock.csv").to_dict("list")
except:
    stock_data = {}
try:    
    stock_records = pd.read_csv("record.csv").to_dict("list")
    for key in stock_records:
        stock_records[key] = [x for x in stock_records[key] if pd.isnull(x) == False]
except:
    stock_records = {}
#print(stock_data)


records = ['1. ', '2. ', '3. ', '4. ', '5. ']


######## Test Model ###########

# folder list for testing 
#folderlist                          =  DRead.get_filelist(  '.')
sess,saver, state, isTrain, rho_eta = Model.TestModel_ConstructGraph( W,W,FSize,PSize,PStride,NumAction )

saver.restore( sess, 'DeepQ' )


def process():
    data = yf.download(input_stock_Entry.get(),period='3mo')

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
        


def predict(output):
    if input_stock_Entry.get() == "":
        return
    output.configure(text='預測中...', bg='white')
    curS = process()
    QAValues    = sess.run  ( rho_eta, feed_dict={ state: curS.reshape(1,W,W), isTrain:False } )
    curA     = np.round  ( QAValues[1].reshape( ( NumAction) ) )
    color = 'black'
    if curA[0]:
        #畫圓
        color = 'red'
        canvas = tk.Canvas(frame1, height=50, width=50)
        canvas.create_oval(20, 30, 40, 50, fill='red')
        canvas.place(x=130,y=12)
        output.configure(text=' 建議做多', bg='red')
        for i in range(5):
            if records[i] == str(i+1) + '. ':
                records[i] =  records[i] + str(datetime.date.today()) + '  ' + input_stock_Entry.get() + ' 建議做多'
                exec('lb_' + str(i+1) + '.configure(text=records[i], fg=color)')
                break
            if i == 4: # 滿了
                records[0] = '1' + records[1][1:]
                records[1] = '2' + records[2][1:]
                records[2] = '3' + records[3][1:]
                records[3] = '4' + records[4][1:]
                records[4] = '5. ' + str(datetime.date.today()) + '  ' + input_stock_Entry.get() + ' 建議做多'
                for j in range(5):
                    exec('lb_' + str(j+1) + '.configure(text=records[j], fg=color)')
        
    elif curA[1]:
        color = 'black'
        canvas = tk.Canvas(frame1, height=50, width=50)
        canvas.create_oval(20, 30, 40, 50, fill='gray')
        canvas.place(x=130,y=12)
        output.configure(text=' 暫不交易', bg='gray')   
        for i in range(5):
            if records[i] == str(i+1) + '. ':
                records[i] = records[i] + str(datetime.date.today()) + '  ' + input_stock_Entry.get() + '暫不交易'
                exec('lb_' + str(i+1) + '.configure(text=records[i], fg=color)')
                break
            if i == 4: # 滿了
                records[0] = '1' + records[1][1:]
                records[1] = '2' + records[2][1:]
                records[2] = '3' + records[3][1:]
                records[3] = '4' + records[4][1:]
                records[4] = '5. ' + str(datetime.date.today()) + '  ' + input_stock_Entry.get() + '暫不交易' 
                for j in range(5):
                    exec('lb_' + str(j+1) + '.configure(text=records[j], fg=color)')
    elif curA[2]:
        color = 'green'
        canvas = tk.Canvas(frame1, height=50, width=50)
        canvas.create_oval(20, 30, 40, 50, fill='green')
        canvas.place(x=130,y=12)
        output.configure(text=' 建議做空', bg='green')
        for i in range(5):
            if records[i] == str(i+1) + '. ':
                records[i] = records[i] + str(datetime.date.today()) + '  ' + input_stock_Entry.get() + '建議做空'
                exec('lb_' + str(i+1) + '.configure(text=records[i], fg=color)')
                break
            if i == 4: # 滿了
                records[0] = '1' + records[1][1:]
                records[1] = '2' + records[2][1:]
                records[2] = '3' + records[3][1:]
                records[3] = '4' + records[4][1:]
                records[4] = '5. ' + str(datetime.date.today()) + '  ' + input_stock_Entry.get() + '建議做空'
                for j in range(5):
                    exec('lb_' + str(j+1) + '.configure(text=records[j], fg=color)')


def openDirectory():
    dir_path = filedialog.askdirectory(
        title="Select model directory"
    )
    return dir_path

def train():
    df = yf.download(train_input_stock_Entry.get(),start=train_start_date.get(), end=train_end_date.get())
    save_model_directory = openDirectory()
    
    train_close = list(df["Close"])
    train_MACD = project.to_MACD(list(train_close))

    initial_money = int(train_input_init_money.get())
    window_size = 30
    skip = 1
    batch_size = 32
    
    #print(train_MACD_enable_var.get())
    
    agent = project.Agent(
        state_size = window_size,
        window_size = window_size,
        trend = train_close,
        buy_trend = None,
        skip = skip,
        batch_size = batch_size,
        train_MACD = train_MACD,
        buy_MACD = None,
        MACD_enable = train_MACD_enable_var.get()    #  change MACD_enable to 1 if you want to use MACD
    )
    
    agent.train(iterations=int(iterations_Entry.get()), checkpoint=10, initial_money=initial_money)
    agent.save_model(save_model_directory+'/model.ckpt')
    
def test():
    df = yf.download(test_input_stock_Entry.get(),start=test_start_date.get(), end=test_end_date.get())
    
    buy_close = list(df["Close"])
    buy_MACD = project.to_MACD(list(buy_close))

    initial_money = int(test_input_init_money.get())
    window_size = 30
    skip = 1
    batch_size = 32
    
    #print(test_MACD_enable_var.get())
    
    agent = project.Agent(
        state_size = window_size,
        window_size = window_size,
        trend = None,
        buy_trend = buy_close,
        skip = skip,
        batch_size = batch_size,
        train_MACD = None,
        buy_MACD = buy_MACD,
        MACD_enable = test_MACD_enable_var.get()    #  change MACD_enable to 1 if you want to use MACD
    )
    agent.load_model(openDirectory()+'/model.ckpt')
    
    states_buy, states_sell, total_gains, invest = agent.buy(initial_money=initial_money)
    ### Plot
    fig = plt.figure(figsize=(15, 5))
    plt.plot(buy_close, color='r', lw=2.)
    plt.plot(buy_close, '^', markersize=10, color='m', label='buying signal', markevery=states_buy)
    plt.plot(buy_close, 'v', markersize=10, color='k', label='selling signal', markevery=states_sell)
    plt.title('total gains %f, total investment %f%%' % (total_gains, invest))
    plt.legend()
    # plt.savefig(''+name+'.png')  # Save the plot as an image
    plt.show()

def select_model():
    new_stock_model_Label.configure(text=openDirectory())
    print(new_stock_model_Label.cget("text"))

def new_stock(stock_data):
    stock_data[new_stock_name_Entry.get()] = [new_stock_number_Entry.get(), new_stock_model_Label.cget("text"), new_stock_MACD_enable_var.get()]
    pd.DataFrame.from_dict(stock_data, orient='index').transpose().to_csv('stock.csv',index=False)
    stockcombobox['values'] += (new_stock_name_Entry.get(),)
    tk.messagebox.showinfo("new stock","新增成功")    

def show_how_to_buy():
    
    if "未選擇"==stockcombobox.get():
        return

    t = time.time()
    t1 = time.localtime(t)
    t2 = time.strftime('%Y-%m-%d',t1)    

    df = yf.download(stock_data[stockcombobox.get()][0], period='3mo').reset_index()
    
    buy_close = list(df["Close"])
    buy_MACD = project.to_MACD(list(buy_close))

    now_money = float(new_stock_init_money_Entry.get())
    holding_inventory = int(stock_amount_Entry.get())
    window_size = 30
    skip = 1
    batch_size = 32
    
    #print(test_MACD_enable_var.get())
    
    agent = project.Agent(
        state_size = window_size,
        window_size = window_size,
        trend = None,
        buy_trend = buy_close,
        skip = skip,
        batch_size = batch_size,
        train_MACD = None,
        buy_MACD = buy_MACD,
        MACD_enable = stock_data[stockcombobox.get()][2]    #  change MACD_enable to 1 if you want to use MACD
    )
    agent.load_model(stock_data[stockcombobox.get()][1] +'/model.ckpt')
    
    today_data = df.loc[df['Date'] == t2 ]

    
    if not today_data.empty:
        
        today = today_data.index.values[0]
        
        #data_name = stockcombobox.get()

        state = agent.get_buy_state(today)
        action = agent.buy_act(state)

        ### Action == Buy
        if (
            action != 0
            and action <= (agent.action_size_half)
            and now_money >= action * agent.buy_trend[today]
            # and today < (len(self.buy_trend) - self.half_window)
        ):
          suggestion_Label.configure(text="buy %d unit at price %.2f" % (action ,agent.buy_trend[today]))
          #print(
          #  "buy %d unit at price %f"
          #   % (action ,agent.buy_trend[today])
          #)
        ### Action == Sell
        elif (
            action > ( agent.action_size_half )
            and holding_inventory > -1 + action - agent.action_size_half
            # and self.buy_trend[t] > np.mean(inventory)
        ):
          act = action - agent.action_size_half # 賣出數量
          suggestion_Label.configure(text="sell %d unit at price %.2f" % (act ,agent.buy_trend[today]))
          #print(
          #  "sell %d unit at price %f"
          #   % (act ,agent.buy_trend[today]))

        else:
          suggestion_Label.configure(text="不交易")
        
        '''
        ## load_state
        try :
            tf = open("stock/" + data_name + ".json", "r")
            prev_datas = json.load(tf)
            states_buy, states_sell, total_gains, invest = agent.buy_test(initial_money=initial_money, today = today,
            path = "stock/" + data_name,
            end = -1, prev_datas = prev_datas )

        except :
            states_buy, states_sell, total_gains, invest = agent.buy_test(initial_money=initial_money, today = today,
            path = "stock/" + data_name,
            end = len(agent.buy_trend) // 2, prev_datas = 0 )
        '''
    else:
        suggestion_Label.configure(text="今天休市或未開市")
        #print("今天休市或未開市")


    
def newFile(stock_records):
    if record_init_money_Entry.get() and new_record_name_Entry.get():
        if record_init_money_Entry.get().isnumeric():
            stock_records[new_record_name_Entry.get()] = []
            stock_records[new_record_name_Entry.get()] += ["init_money "+record_init_money_Entry.get()]
            recordcombobox['values'] += (new_record_name_Entry.get(),)
            tk.messagebox.showinfo("新增","新增成功，請到下拉選單選取新增的紀錄簿。")
        else:
            tk.messagebox.showinfo("新增-初始資金", "非數字輸入")

def add_record(stock_records):
    #print(stock_records)
    #print(stock_records[recordcombobox.get()])
    if "未選擇"!=recordcombobox.get():
        money_entry = record_change_money_Entry.get()
        if money_entry[0] == '+':
            if money_entry[1:].isnumeric():
                stock_records[recordcombobox.get()] += [record_stock_Entry.get()+" "+ money_entry]
                tk.messagebox.showinfo("增加項目","成功增加項目")
            else:
                tk.messagebox.showinfo("資金變化", "非數字輸入")
        elif money_entry[0] == '-':
            if money_entry[1:].isnumeric():
                stock_records[recordcombobox.get()] += [record_stock_Entry.get()+" "+ money_entry]
                tk.messagebox.showinfo("增加項目","成功增加項目")
            else:
                tk.messagebox.showinfo("資金變化", "非數字輸入")
        else:
            if money_entry.isnumeric():
                stock_records[recordcombobox.get()] += [record_stock_Entry.get()+" "+ money_entry]
                tk.messagebox.showinfo("增加項目","成功增加項目")
            else:
                tk.messagebox.showinfo("資金變化", "非數字輸入")
    #print(stock_records[recordcombobox.get()])
    #stock_records[recordcombobox.get()] = record_stock_Entry.get+ " " +record_change_money_Entry.get()# = record.add(stock_records,[ record_stock_Entry.get(), int(record_change_money_Entry.get()) ])

def compute_result(stock_records):
    if "未選擇"!=recordcombobox.get():
        init_money = float(stock_records[recordcombobox.get()][0].split()[-1])
        final_money = 0
        for i in stock_records[recordcombobox.get()]:
            final_money += float(i.split()[-1])
        
        color = 'black'
        if final_money/init_money - 1 < 0:
            color = 'green'
        elif final_money/init_money - 1 > 0:
            color = 'red' 
        result_Label.configure(text=f'初始資金: {init_money}, 目前資金: {final_money}, 損益率: {(final_money/init_money-1)*100:.2f}%', fg=color)

def save_record(stock_records):
    if "未選擇"!=recordcombobox.get():
        pd.DataFrame.from_dict(stock_records, orient='index').transpose().to_csv('record.csv',index=False,encoding='utf-8-sig')
        tk.messagebox.showinfo("存檔","存檔成功")

    
# 第1步，实例化object，建立窗口window
window = tk.Tk()

# 第2步，给窗口的可视化起名字
window.title('股票預測')

# 第3步，设定窗口的大小(长 * 宽)
window.geometry('500x350')  # 这里的乘是小x

notebook = ttk.Notebook(window)

# 第5步，创建一个主frame1，长在主window窗口上
frame1 = tk.Frame(notebook)
notebook.add(frame1, text="信號燈")

output = tk.Label(frame1, text='尚未預測', bg='gray' , font=('Arial 20'))

input_stock_Entry = tk.Entry(frame1, font=('Arial 20'), width=10)
lb_1 = tk.Label(frame1, text='股票代號:', font=('Arial 20')) 

btn = tk.Button(frame1,text="交易建議", font=('Arial 20'), command=lambda : predict(output), bg='yellow')


output.pack(pady=32)

lb_1.place(x=20, y=120)
input_stock_Entry.pack(pady=20)

btn.pack()

###############################

frame2 = tk.Frame(notebook)
notebook.add(frame2, text="train and test")

stock_Label = tk.Label(frame2, text="股票代號:", font=('Arial 15'))
init_money_Label = tk.Label(frame2, text="初始資金:", font=('Arial 15'))
start_date_Label = tk.Label(frame2, text="開始日期:", font=('Arial 15'))
end_date_Label = tk.Label(frame2, text="結束日期:", font=('Arial 15'))
iterations_Label = tk.Label(frame2, text="訓練次數:", font=('Arial 15'))

test_stock_Label = tk.Label(frame2, text="股票代號:", font=('Arial 15'))
test_init_money_Label = tk.Label(frame2, text="初始資金:", font=('Arial 15'))
test_start_date_Label = tk.Label(frame2, text="開始日期:", font=('Arial 15'))
test_end_date_Label = tk.Label(frame2, text="結束日期:", font=('Arial 15'))

train_input_stock_Entry = tk.Entry(frame2 , font=('Arial 20'), width=10)
train_input_init_money = tk.Entry(frame2 , font=('Arial 20'), width=10)
train_start_date = tk.Entry(frame2 , font=('Arial 20'), width=10)
train_end_date = tk.Entry(frame2 , font=('Arial 20'), width=10)
iterations_Entry = tk.Entry(frame2 , font=('Arial 20'), width=10)

test_input_stock_Entry = tk.Entry(frame2 , font=('Arial 20'), width=10)
test_input_init_money = tk.Entry(frame2 , font=('Arial 20'), width=10)
test_start_date = tk.Entry(frame2 , font=('Arial 20'), width=10)
test_end_date = tk.Entry(frame2 , font=('Arial 20'), width=10)

btn_trian = tk.Button(frame2,text="train", font=('Arial 20') ,command=lambda : train())


btn_test = tk.Button(frame2,text="test",font=('Arial 20'),command=lambda : test())

stock_Label.grid(row = 0,column = 0)
init_money_Label.grid(row = 1,column = 0)
start_date_Label.grid(row = 2,column = 0)
end_date_Label.grid(row = 3,column = 0)
iterations_Label.grid(row = 4,column = 0)

train_input_stock_Entry.grid(row = 0,column = 1)
train_input_init_money.grid(row = 1,column = 1)
train_start_date.grid(row = 2,column = 1)
train_end_date.grid(row = 3,column = 1)
iterations_Entry.grid(row = 4,column = 1)

train_MACD_enable_var = tk.IntVar()
train_MACD_enable = tk.Checkbutton(frame2, text='MACD_enable',variable=train_MACD_enable_var, onvalue=1, offvalue=0)    # 传值原理类似于radiobutton部件

test_stock_Label.grid(row = 0,column = 2)
test_init_money_Label.grid(row = 1,column = 2)
test_start_date_Label.grid(row = 2,column = 2)
test_end_date_Label.grid(row = 3,column = 2)

test_input_stock_Entry.grid(row = 0,column = 3)
test_input_init_money.grid(row = 1,column = 3)
test_start_date.grid(row = 2,column = 3)
test_end_date.grid(row = 3,column = 3)

test_MACD_enable_var = tk.IntVar()
test_MACD_enable = tk.Checkbutton(frame2, text='MACD_enable',variable=test_MACD_enable_var, onvalue=1, offvalue=0)    # 传值原理类似于radiobutton部件

train_MACD_enable.grid(row = 5,column = 1)
test_MACD_enable.grid(row = 4,column = 3)

btn_trian.grid(row = 6,column = 1)
btn_test.grid(row = 6,column = 3)

#################################
frame5 = tk.Frame(notebook)
notebook.add(frame5, text="實戰")

new_stock_Label = tk.Label(frame5, text="新增股票", font=('Arial 15'))

new_stock_name_Label = tk.Label(frame5, text="項目名稱:", font=('Arial 15'))
new_stock_name_Entry = tk.Entry(frame5 , font=('Arial 20'))
new_stock_number_Label = tk.Label(frame5, text="股票代號:", font=('Arial 15'))
new_stock_number_Entry = tk.Entry(frame5 , font=('Arial 20'))
bnt_new_stock_model = tk.Button(frame5,text="選擇model", font=('Arial 10') ,command=lambda : select_model())
new_stock_model_Label = tk.Label(frame5, font=('Arial 15'))
new_stock_MACD_enable_var = tk.IntVar()
new_stock_MACD_enable = tk.Checkbutton(frame5, text='MACD_enable',variable=new_stock_MACD_enable_var, onvalue=1, offvalue=0)
bnt_new_stock = tk.Button(frame5,text="新增股票", font=('Arial 10') ,command=lambda : new_stock(stock_data))

new_stock_init_money_Label = tk.Label(frame5, text="目前資金:", font=('Arial 15'))
new_stock_init_money_Entry = tk.Entry(frame5 , font=('Arial 20'))

stock_amount_Label = tk.Label(frame5, text="股票數量:", font=('Arial 15'))
stock_amount_Entry = tk.Entry(frame5 , font=('Arial 20'))

stockcombobox = ttk.Combobox(frame5, state='readonly')
stockcombobox['values'] = ["未選擇"]+list(stock_data.keys())
stockcombobox.current(0)

bnt_suggestion = tk.Button(frame5,text="買賣建議", font=('Arial 10') ,command=lambda : show_how_to_buy())

suggestion_Label = tk.Label(frame5, font=('Arial 15'))

new_stock_Label.grid(row = 0,column = 1)

new_stock_name_Label.grid(row = 1,column = 0)
new_stock_name_Entry.grid(row = 1,column = 1)

new_stock_number_Label.grid(row = 2,column = 0)
new_stock_number_Entry.grid(row = 2,column = 1)

bnt_new_stock_model.grid(row = 3,column = 0)
new_stock_model_Label.grid(row = 3,column = 1, columnspan=5)

new_stock_MACD_enable.grid(row = 4,column = 0)
bnt_new_stock.grid(row = 4,column = 1)

stockcombobox.grid(row = 5,column = 0)

new_stock_init_money_Label.grid(row = 6,column = 0)
new_stock_init_money_Entry.grid(row = 6,column = 1)

stock_amount_Label.grid(row = 7,column = 0)
stock_amount_Entry.grid(row = 7,column = 1)


bnt_suggestion.grid(row = 8,column = 0)
suggestion_Label.grid(row = 8,column = 1)

#################################
frame3 = tk.Frame(notebook)
notebook.add(frame3, text="信號燈紀錄")


lb_1 = tk.Label(frame3, text=records[0] + "暫無紀錄", font=('Arial 15'))
lb_2 = tk.Label(frame3, text=records[1] + "暫無紀錄", font=('Arial 15'))
lb_3 = tk.Label(frame3, text=records[2] + "暫無紀錄", font=('Arial 15'))
lb_4 = tk.Label(frame3, text=records[3] + "暫無紀錄", font=('Arial 15'))
lb_5 = tk.Label(frame3, text=records[4] + "暫無紀錄", font=('Arial 15'))

lb_1.pack(pady=10)
lb_2.pack(pady=10)
lb_3.pack(pady=10)
lb_4.pack(pady=10)
lb_5.pack(pady=10)

###############################
frame4 = tk.Frame(notebook)
notebook.add(frame4, text="記帳功能")

recordcombobox = ttk.Combobox(frame4, state='readonly', font=('Arial 15'), width=10)
recordcombobox['values'] = ["未選擇"]+list(stock_records.keys())
recordcombobox.current(0)

record_stock_Label = tk.Label(frame4, text="項目名稱:", font=('Arial 15'))
record_stock_Entry = tk.Entry(frame4 , font=('Arial 18'), width=10)

record_change_money_Label = tk.Label(frame4, text="資金變化:", font=('Arial 15'))
record_change_money_Entry = tk.Entry(frame4 , font=('Arial 18'), width=10)


btn_add = tk.Button(frame4,text="增加項目", font=('Arial 10') ,command=lambda : add_record(stock_records) )
btn_save = tk.Button(frame4,text="存入檔案", font=('Arial 10') ,command=lambda : save_record(stock_records))
btn_result = tk.Button(frame4,text="計算損益", font=('Arial 10') ,command=lambda : compute_result(stock_records) )
result_Label = tk.Label(frame4, font=('Arial 15'))

new_record_Label = tk.Label(frame4, text="新增紀錄簿", font=('Arial 15'))
new_record_name_Label = tk.Label(frame4, text="紀錄名稱:", font=('Arial 15'))
new_record_name_Entry = tk.Entry(frame4 , font=('Arial 18'), width=10)
record_init_money_Label = tk.Label(frame4, text="初始資金:", font=('Arial 15'))
record_init_money_Entry = tk.Entry(frame4 , font=('Arial 18'), width=10)

btn_new = tk.Button(frame4,text="新增", font=('Arial 10') ,command=lambda : newFile(stock_records))

Combobox_select = tk.Label(frame4, text="選擇紀錄:", font=('Arial 15'))
separ = ttk.Separator(frame4, orient= tk.HORIZONTAL)

Combobox_select.grid(row = 0, column = 0)
recordcombobox.grid(row = 0, column = 1)

record_stock_Label.grid(row = 1, column = 0)
record_stock_Entry.grid(row = 1, column = 1)

record_change_money_Label.grid(row = 2, column = 0)
record_change_money_Entry.grid(row = 2, column = 1)

btn_add.grid(row = 3, column = 0)
btn_result.grid(row = 3, column = 1)
btn_save.grid(row = 3, column = 2)

result_Label.grid(row = 4, column = 0, columnspan=4)

# separ.grid(row=5, ipadx=80, pady=10)

new_record_Label.grid(row = 6, column = 1)

new_record_name_Label.grid(row = 7, column = 0)
new_record_name_Entry.grid(row = 7, column = 1)
record_init_money_Label.grid(row = 8, column = 0)
record_init_money_Entry.grid(row = 8, column = 1)

btn_new.grid(row = 9, column = 1)


notebook.pack(expand=1, fill="both")

# 第8步，主窗口循环显示
window.mainloop()



