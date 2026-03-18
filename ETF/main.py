import os
import time 
import numpy as np
import pandas as pd
import akshare as ak
import quantstats as qs
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import warnings
warnings.filterwarnings('ignore')

import akshare_proxy_patch

akshare_proxy_patch.install_patch(
    "101.201.173.125",
    # 免费版为空，付费版填入具体TOKEN
    auth_token="",
    retry=30,
    # 封控的域名列表，可动态调整
    hook_domains=[
      "fund.eastmoney.com",
      "push2.eastmoney.com",
      "push2his.eastmoney.com",
      "emweb.securities.eastmoney.com",
    ],
)

if __name__ == "__main__":
    # 510300：沪深300ETF，代表大盘
    # 510500：中证500ETF，代表小盘
    # 510880：红利ETF，代表价值
    # 159915：创业板ETF，代表成长
    code_list = ['510300', '510500', '510880', '159915']
    start_date = '20150101'
    end_date = '20250828'

    df_list = []
    for code in code_list:
        print(f'正在获取[{code}]行情数据...')
        # adjust：""-不复权、qfq-（前复权）、hfq-后复权
        df = ak.fund_etf_hist_em(symbol=code, period='daily', 
            start_date=start_date, end_date=end_date, adjust='hfq')
        df.insert(0, 'code', code)
        df_list.append(df)
        time.sleep(3)
    print('数据获取完毕！')

    all_df = pd.concat(df_list, ignore_index=True)
    data = all_df.pivot(index='日期', columns='code', values='收盘')[code_list]
    data.index = pd.to_datetime(data.index)
    data = data.sort_index()
    data.head(10)