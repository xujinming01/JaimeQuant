import unicodedata
import akshare as ak
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
import concurrent.futures

import factors
import filters

warnings.filterwarnings("ignore")

# ---------------- 代理配置区域 ----------------
import akshare_proxy_patch

akshare_proxy_patch.install_patch(
    "101.201.173.125",
    auth_token="",
    retry=30,
    hook_domains=[
      "fund.eastmoney.com",
      "push2.eastmoney.com",
      "push2his.eastmoney.com",
      "emweb.securities.eastmoney.com",
    ],
)
# ----------------------------------------------

# ----------------- 实盘推送配置区域 -----------------
ETF_DICT = {
    # '510880': '红利ETF华泰柏瑞',
    # '159915': '创业板ETF易方达',
    '513100': '纳指ETF',
    '518880': '黄金ETF华安',
    '512890': '红利低波ETF华泰柏瑞',
    '588000': '科创50ETF',
    '159949': '创业板50ETF华安',
    '512100': '中证1000ETF南方',
    '563300': '中证2000ETF华泰柏瑞',
    '159985': '豆粕ETF',
}
MOMENTUM_WINDOW = 20

# 如果用到 RSRS 高级版(z_window=300)，这里要设为 400 确保数据长度足够
LOOKBACK_DAYS = MOMENTUM_WINDOW + 10
# ----------------------------------------------

def get_visual_width(text):
    """
    计算单个字符串的视觉宽度。
    中文（全角）占2个宽度，英文/数字（半角）占1个宽度。
    """
    return sum(2 if unicodedata.east_asian_width(c) in ('F', 'W') else 1 for c in text)

def align_text(text, target_width):
    """
    计算当前字符串的视觉宽度，并用空格右侧补齐到 target_width。
    """
    visual_width = get_visual_width(text)
    padding_length = max(0, target_width - visual_width)
    return text + ' ' * padding_length

def get_realtime_daily_k(symbol, lookback_days=400):
    """
    获取包含今天（截至14:50）的最新日线数据
    采用: 历史日线 + 今日分钟线合成 的方式
    """
    # 按照 1.5 倍日历日推算，外加 20 天节假日缓冲，足够覆盖所需的交易日
    buffer_days = int(lookback_days * 1.5) + 20
    start_date_str = (datetime.now() - timedelta(days=buffer_days)).strftime("%Y%m%d")
    end_date_str = datetime.now().strftime("%Y%m%d")
    
    daily_df = ak.fund_etf_hist_em(
        symbol=symbol, 
        period="daily", 
        start_date=start_date_str, 
        end_date=end_date_str, 
        adjust="qfq"
    )
    daily_df['日期'] = pd.to_datetime(daily_df['日期']).dt.date
    daily_df = daily_df.set_index('日期')[['收盘', '最高', '最低']]
    
    # 2. 获取今日 1分钟线，合成今日（截至14:50）的日线数据
    today_date = datetime.now().date()
    
    try:
        start_time = f"{today_date} 09:30:00"
        end_time = f"{today_date} 14:50:00"  # 👈 直接在接口端卡死结束时间
        
        min_df = ak.fund_etf_hist_min_em(
            symbol=symbol, 
            period="1", 
            adjust="", 
            start_date=start_time, 
            end_date=end_time
        )
        
        if not min_df.empty:
            # 取最后一行，如果是 14:50 拿到的就是 14:50；如果没有，自动就是 14:49 或更早的最新价
            latest_bar = min_df.iloc[-1]
            today_close = latest_bar['收盘']
            print(f"✅ {symbol} 今日最新价（截至{latest_bar['时间']}）: {today_close}")
            
            # 最高和最低价必须取当天 09:30 到 14:50 期间的极值，否则日线 K 线特征会失真
            today_high = min_df['最高'].max()
            today_low = min_df['最低'].min()
            
            # 覆盖或追加今日的数据
            if daily_df.index[-1] == today_date:
                daily_df.loc[today_date, '收盘'] = today_close
                daily_df.loc[today_date, '最高'] = today_high
                daily_df.loc[today_date, '最低'] = today_low
            else:
                new_row = pd.DataFrame({
                    '收盘': [today_close],
                    '最高': [today_high],
                    '最低': [today_low]
                }, index=[today_date])
                daily_df = pd.concat([daily_df, new_row])

    except Exception as e:
        print(f"⚠️ {symbol} 今日分钟线获取失败或尚未开盘，使用原有日线兜底: {e}")

    daily_df = daily_df.sort_index()
    return daily_df.tail(lookback_days)

def fetch_worker(code):
    try:
        df = get_realtime_daily_k(code, lookback_days=LOOKBACK_DAYS)
        return code, df
    except Exception as e:
        print(f"❌ 获取 {code} 数据彻底失败: {e}")
        return code, None

def main():
    
    print(f"🕒 正在实时抓取数据并合成今日(截至14:50) 的 K 线特征...")
    
    close_list, high_list, low_list = [], [],[]
    
    # max_workers=5 表示同时发起 5 个请求。对于东方财富的接口，5~8 是比较安全的并发数，太高容易被封 IP
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # executor.map 会保持返回结果与传入字典键的顺序一致
        results = executor.map(fetch_worker, ETF_DICT.keys())
        
    for code, df in results:
        if df is not None and not df.empty:
            close_list.append(df[['收盘']].rename(columns={'收盘': code}))
            high_list.append(df[['最高']].rename(columns={'最高': code}))
            low_list.append(df[['最低']].rename(columns={'最低': code}))
        else:
            print(f"⚠️ 跳过 {ETF_DICT[code]} ({code}) 的拼接")
        
    prices = pd.concat(close_list, axis=1).ffill()
    highs = pd.concat(high_list, axis=1).ffill()
    lows = pd.concat(low_list, axis=1).ffill()
    
    # ============== 模块化调用 ==============
    print(f"🧮 正在计算策略得分...")
    
    # 1. 直接传入 DataFrame 给外部独立模块计算
    factor = factors.calc_pure_momentum(prices, window=MOMENTUM_WINDOW) 
    
    # # 2. 调用过滤器 (如果有被过滤掉的，可以直接把分设为极小值或 NaN)
    # drop_safe_mask = filters.filter_recent_drop(prices, 0.05)
    # factor = factor.where(drop_safe_mask, np.nan) 
    # ============== 模块化调用结束 ==============
    
    # 获取今天（DataFrame 最后一行）的各 ETF 得分
    today_scores = factor.iloc[-1]
    
    print("="*45)
    print(f"📈 今日 ({prices.index[-1].strftime('%Y-%m-%d')}) 最终信号排名:")
    print("="*45)
    
    # 从高到低排序
    ranked = today_scores.sort_values(ascending=False)
    for code, score in ranked.items():
        # 使用自定义函数对齐名称
        max_name_width = max(get_visual_width(name) for name in ETF_DICT.values())
        name_aligned = align_text(ETF_DICT[code], max_name_width+1)
        print(f" {name_aligned} ({code}): {score*100:6.2f}%")
    print("-" * 45)
    
    # 提取第一名
    best_code = ranked.index[0]
    best_score = ranked.iloc[0]

    if best_score > 0:
        print(f"🎯 实盘建议操作: \n👉 满仓持有 / 买入 【{ETF_DICT[best_code]} ({best_code})】")
    else:
        print(f"⚠️ 预警: 所有标的动量均为负数！\n👉 实盘建议操作: 空仓，或买入避险资产 (如 161119 货币/债券)")
    print("="*45)

    return prices, ranked, best_code, best_score

if __name__ == "__main__":
    main()