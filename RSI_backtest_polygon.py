# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 11:13:27 2022

@author: USER
"""

import streamlit as st 
import pandas_ta as ta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests 
import pandas as pd
from datetime import date, datetime, timedelta, time


API = "AI73Oy1KnUYUfHAKsOJLxSmVz9dWFC95"
deflaut_date = date.today() - timedelta(100)
min_date =  date.today() - timedelta(365)
max_date =  date.today()- timedelta(1)
# st.set_page_config(layout="wide")


def drawndown(trade_result_log):
    max_return, list_drawn_down  = [100], []
    for index,row in trade_result_log.iterrows():
        max_return.append(row["accumulated_return"])
        drawn_down = (row["accumulated_return"] - max(max_return))/max(max_return)
        list_drawn_down.append(drawn_down)
    return list_drawn_down


def get_data(symbol, end_date, day_range, interval = 15):
  url = "https://api.polygon.io/v2/"
  symbol = symbol.upper()

  time_frame = "minute"
  limit = 40000
  sort = "desc"

  end_time = datetime.combine(end_date, datetime.min.time()) # combin to timestamp object
  start_time = end_time - timedelta(days = day_range)

  print(f"Downloading {start_time} to {end_time} {symbol} Data")

  start_time = int(start_time.timestamp() * 1000) # convert to ms
  end_time = int(end_time.timestamp() * 1000)  

  request_url = f"{url}aggs/ticker/{symbol}/range/{interval}/{time_frame}/{start_time}/{end_time}?adjusted=true&sort={sort}&limit={limit}&apiKey={API}"

  data = requests.get(request_url).json()
  if "results" in data: 
    return data["results"]
  else: 
    print("no data")
    pass 


st.title('RSI backtest')
ticker = st.sidebar.text_input("Enter ticker here 👇", value="AAPL")
time_interval_selection = ["5 mins", "15 mins", "1 hour", "4 hour"]
time_interval_select = st.sidebar.selectbox("Select Time Interval", time_interval_selection)

if time_interval_select == "5 mins":
    time_interval = 5 

elif time_interval_select == "15 mins":
    time_interval = 15 
    
elif time_interval_select == "1 hour":
    time_interval = 60 
    
elif time_interval_select == "4 hour":
    time_interval = 240 



start_date_input = st.sidebar.date_input("start date (Max length 1 year for this version)", value=deflaut_date, min_value = min_date, max_value=max_date)
rsi_length = st.sidebar.slider("RSI Length", min_value=1, max_value=30, value=6)

# if st.sidebar.button('Get Data'):
interval = int(time_interval)
end_date = date.today()
end_date_display = date.today()
start_date = start_date_input
day_range = (end_date - start_date).days #datatime to days
list_bars, bar = [],[]
end_date = date.today()
run_period = 60 
if day_range <run_period:
    list_bars = get_data(ticker, end_date, day_range, interval =interval)
else:
    while day_range >run_period:
        bar = get_data(ticker, end_date, day_range = run_period, interval =interval)
        list_bars += bar
        day_range -= run_period
        end_date -= timedelta(days = run_period)

    bar = get_data(ticker, end_date, day_range, interval =interval)
    list_bars += bar
    
    df = pd.DataFrame(list_bars)
    df["datetime"] = pd.to_datetime(df["t"], unit="ms")
    df.set_index("datetime", inplace=True)
    df = df [["o","h","l","c","v","n"]]
    df.columns = ["Open","High","Low","Close","Volume","Transaction"]
    
    # #convert Time zone to E.T time 
    # eastern = pytz.timezone('US/Eastern')
    # df.index = df.index.tz_localize(pytz.utc).tz_convert(eastern)

df.sort_index(ascending=True, inplace=True)
df["rsi"] = ta.rsi(df.Close, length=rsi_length)
df.dropna(inplace=True)
st.write(f"Testing data from {start_date} to {end_date_display} total {len(df)} rows data")
with st.expander("Click to expand", expanded=False):
    st.write(f"Total {len(df)} rows data")
    st.dataframe(df)

#RSI Signal 
col1, col2, col3, col4 =st.columns([1.3,1,1.3,1])
with col1:
    st.markdown("RSI crossover down (buy):")
with col2:
    rsi_under = st.number_input("RSI Under:",min_value=10, max_value=50, value=30, label_visibility="collapsed")
with col3:
    st.markdown("RSI crossover up (sell):")
with col4:
    rsi_over = st.number_input("3", min_value=50, max_value=90, value=70, label_visibility="collapsed")

#calculate rsi signal
conditions = [(df["rsi"]<rsi_under)&(df["rsi"].shift()>rsi_under), (df["rsi"]>rsi_over)&(df["rsi"].shift()<rsi_over)]
actions = ["buy","sell"]
df["signal"] = np.select(conditions, actions) 
df["Adjusted_signal"] = df["signal"].shift() #buy or sell at next open 
df.dropna(inplace=True)

#check crossover
position = False 
list_open_date, list_close_date = [],[]
list_open_price, list_close_price = [],[]
list_order_type = []

for index,row in df.iterrows():
    if not position:
        if row["Adjusted_signal"] != "0":
            order_type = row["Adjusted_signal"]
            list_open_date.append(index)
            list_open_price.append(row["Open"])
            list_order_type.append(order_type)
            position = True
    if position:
        if (row["Adjusted_signal"] != "0") & (row["Adjusted_signal"] != order_type):
            list_close_date.append(index)
            list_close_price.append(row["Open"])  
            position = False

if position: #for position not close at the end 
    list_close_date.append(df.index[-1])
    list_close_price.append(df["Open"].iloc[-1])  
    
trade_result_log = pd.DataFrame({"open_date":list_open_date, "close_date":list_close_date, 
                   "open_price":list_open_price,"close_price":list_close_price,
                  "order_type":list_order_type}).set_index("open_date")

rsi_result = pd.concat([df, trade_result_log ], axis = 1)

#Calculate result static  


conditions = [(trade_result_log["order_type"] == "buy"), (trade_result_log["order_type"] == "sell")]
actions = [((trade_result_log["close_price"] - trade_result_log["open_price"])/trade_result_log["open_price"]), 
           (trade_result_log["open_price"] - trade_result_log["close_price"])/trade_result_log["open_price"]]
trade_result_log["trade_return"] = np.select(conditions, actions) 

accumulated_return = 100
list_accumulated_return = []  
for index,row in trade_result_log.iterrows():
    accumulated_return = (1+row["trade_return"]) *accumulated_return
    list_accumulated_return.append(accumulated_return)

trade_result_log["accumulated_return"] = list_accumulated_return

final_result = trade_result_log["accumulated_return"][-1] -100
testing_period = (df.index[-1] - df.index[0])/np.timedelta64(1, 'D')
number_of_trade = (len(trade_result_log))

win_rate = (len(trade_result_log[trade_result_log["trade_return"] >0])/number_of_trade)
loss_rate = (len(trade_result_log[trade_result_log["trade_return"] <0])/number_of_trade)
win_loss_ratiio = win_rate/loss_rate

best_trade = max(trade_result_log["trade_return"])*100
worst_trade = min(trade_result_log["trade_return"])*100
Longest_trade_holding = max((trade_result_log["close_date"] - trade_result_log.index)/np.timedelta64(1, 'D'))
reward_ratio = sum(trade_result_log.loc[trade_result_log['trade_return'] > 0]["trade_return"])/len(trade_result_log[trade_result_log["trade_return"] >0])
risk_ratio = sum(trade_result_log.loc[trade_result_log['trade_return'] < 0]["trade_return"])/len(trade_result_log[trade_result_log["trade_return"] <0])
risk_reward_ratio = reward_ratio/(-risk_ratio)
max_dawndown = min(drawndown(trade_result_log))*100






tab1, tab2, tab3, tab4 = st.tabs(["Static", "Chart", "Trade Result Log", "Singal Log"])


with tab1:
    st.header("Total")
    col1, col2, col3= st.columns([1,1,1])
    with col1:
        st.metric(
            "Total Result",
            f"{final_result:.2f}%")
        st.metric(
            "Maximum Drawdown",
            f"{max_dawndown:.2f}%")
    with col2:
        st.metric(
            "Total Win/Loss Rate",
            f"{win_loss_ratiio:.2f}")
        st.metric(
            "Testing Period (Days)",
            f"{testing_period:.1f}")     
    with col3:
        st.metric(
           "Risk/Reward Ratio",
           f"{risk_reward_ratio:.2f}")
        st.metric(
           "Number of Trade",
           f"{number_of_trade:.0f}")

    st.header("Single Trade")
    col1, col2, col3= st.columns([1,1,1])
    with col1:
        st.metric(
            "Best Trade",
            f"{best_trade:.2f}%")

    with col2:
        st.metric(
            "Worst Trade",
            f"{worst_trade:.2f}%")
   
    with col3:
        st.metric(
           "Longest Trade Duration (Days)",
           f"{Longest_trade_holding:.1f}")


    st.title("Backtest Stragey ")
    with st.expander("Open Position", expanded=False):
        st.text(f"Buy on next open price, once RIS crossover down {rsi_under}")
        st.text(f"Sell on next open price, once RIS crossover up {rsi_over}")
    
    with st.expander("Close Position, Stop loss, Take Profit", expanded=False):
    
        st.text("Once RSI crossover opposite signal")
        st.text("No Stop Loss")
        st.text("No Take Profit")
    
    with st.expander("Trading Fee", expanded=False):
        st.text("No Trading Fee")
                
with tab2:
    # Create subplots and mention plot grid size
    fig_ticker = make_subplots(rows=2, cols=1, shared_xaxes=True, 
               vertical_spacing=0.03, subplot_titles=('OHLC', 'Volume'), 
               row_width=[0.2, 0.7])
    
    # fig_ticker = go.Figure()
    fig_ticker.add_trace(go.Candlestick(x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"]) )    # Plot OHLC on 1st row
    fig_ticker.update_layout(xaxis_rangeslider_visible=False) # Do not show OHLC's rangeslider plot 
    st.plotly_chart(fig_ticker)   

    

    fig = px.line(
          trade_result_log,
          x="close_date",
          y="accumulated_return")
    st.plotly_chart(fig) 
    #st.dataframe(trade_result_log)        
with tab3:
    st.dataframe(trade_result_log) 
with tab4:
    rsi_result = pd.concat([df, trade_result_log ], axis = 1)
    st.dataframe(rsi_result) 