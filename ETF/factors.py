import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

def calc_pure_momentum(prices, window=20):
    """计算过去 N 天的动量涨幅"""
    print(f"🧮 计算因子: {window}日纯动量...")
    factor_df = prices / prices.shift(window) - 1.0
    return factor_df

def calc_atr_dynamic_score(prices, highs, lows, lb_min=20, lb_max=60, vol_short_len=20, vol_long_len=60, ratio_cap=0.9):
    """基于真实 ATR 计算动态回溯窗口得分"""
    print(f"🧮 计算因子: 基于真实ATR的动态窗口回归得分...")
    
    prev_close = prices.shift(1)
    
    tr1 = highs - lows
    tr2 = (highs - prev_close).abs()
    tr3 = (lows - prev_close).abs()
    
    # 取三者的最大值为当天的 TR
    tr = pd.DataFrame(
        np.maximum(tr1.values, np.maximum(tr2.values, tr3.values)),
        index=prices.index,
        columns=prices.columns
    )
    
    # 计算短长周期的 ATR
    atr_short = tr.rolling(vol_short_len).mean()
    atr_long = tr.rolling(vol_long_len).mean()
    
    # 计算动态窗口大小
    vol_ratio = atr_short / atr_long
    vol_ratio_capped = vol_ratio.clip(upper=ratio_cap)
    lookback_df = np.floor(lb_min + (lb_max - lb_min) * (1 - vol_ratio_capped)).fillna(lb_max).astype(int)
    
    factor_df = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    
    # 计算动态窗口的线性回归得分
    def fast_score(y):
        if len(y) < 2 or y[0] == 0: return np.nan
        y_norm = y / y[0]
        x = np.arange(1, len(y) + 1)
        cov = np.cov(x, y_norm, ddof=0)[0, 1]
        var_x = np.var(x, ddof=0)
        if var_x == 0: return np.nan
        slope = cov / var_x
        r_squared = np.corrcoef(x, y_norm)[0, 1] ** 2
        return slope * r_squared
        
    for code in prices.columns:
        prices_arr = prices[code].values
        lookback_arr = lookback_df[code].values
        scores = np.full(len(prices_arr), np.nan)
        
        for i in range(lb_max, len(prices_arr)):
            curr_lb = lookback_arr[i]
            window_prices = prices_arr[i - curr_lb + 1 : i + 1]
            scores[i] = fast_score(window_prices)
            
        factor_df[code] = scores
        
    return factor_df

def calc_trend_score(prices, window=25):
    """基于线性回归斜率和决定系数 R2 的趋势得分"""
    print(f"🧮 计算因子: {window}日线性回归趋势得分 (Slope * R2)...")
    idx = pd.Series(np.arange(len(prices)), index=prices.index)

    # 计算滚动窗口内的协方差和相关系数
    rolling_cov = prices.rolling(window).cov(idx)
    rolling_corr = prices.rolling(window).corr(idx)
    
    # 计算斜率和 R2
    var_x = window * (window + 1) / 12  
    slope_raw = rolling_cov / var_x
    r_squared = rolling_corr ** 2
    
    # 斜率标准化：除以窗口内第一个价格的水平，得到相对斜率
    p0 = prices.shift(window - 1) 
    slope_norm = slope_raw / p0
    factor_df = (slope_norm * r_squared).replace([np.inf, -np.inf], np.nan)
    
    return factor_df

def calc_rsrs(highs, lows, window=18):
    """标准 RSRS 因子 (Resistance Support Relative Strength)"""
    print(f"🧮 计算因子: {window}日 RSRS 阻力支撑强弱指标...")
    mean_x = lows.rolling(window).mean()
    mean_y = highs.rolling(window).mean()
    mean_xy = (lows * highs).rolling(window).mean()
    mean_x2 = (lows ** 2).rolling(window).mean()
    mean_y2 = (highs ** 2).rolling(window).mean()
    
    cov_xy = mean_xy - mean_x * mean_y
    var_x = mean_x2 - mean_x ** 2
    var_y = mean_y2 - mean_y ** 2
    
    var_x = var_x.replace(0, np.nan)
    var_y = var_y.replace(0, np.nan)
    
    slope = cov_xy / var_x
    r_squared = (cov_xy ** 2) / (var_x * var_y)
    rsrs_score = slope * r_squared

    return rsrs_score

def calc_rsrs_advanced(highs, lows, window=16, z_window=300):
    """高级 RSRS 因子 (Right-Skewed Standard Score)"""
    print(f"🧮 计算因子: {window}日 RSRS 右偏标准分 (Z观察期: {z_window}日)...")

    # 计算基本的 RSRS 斜率和 R2
    mean_x = lows.rolling(window).mean()
    mean_y = highs.rolling(window).mean()
    mean_xy = (lows * highs).rolling(window).mean()
    mean_x2 = (lows ** 2).rolling(window).mean()
    mean_y2 = (highs ** 2).rolling(window).mean()
    
    cov_xy = mean_xy - mean_x * mean_y
    var_x = mean_x2 - mean_x ** 2
    var_y = mean_y2 - mean_y ** 2
    
    var_x = var_x.replace(0, np.nan)
    var_y = var_y.replace(0, np.nan)
    
    beta = cov_xy / var_x
    r_squared = (cov_xy ** 2) / (var_x * var_y)
    
    # 计算标准分：（beta - beta_mean）/ beta_std
    beta_mean = beta.rolling(z_window, min_periods=20).mean()
    beta_std = beta.rolling(z_window, min_periods=20).std()
    
    beta_std = beta_std.replace(0, np.nan) 
    std_score = (beta - beta_mean) / beta_std
    
    # 计算修正标准分：R2 * 标准分，进一步放大高 R2 时的得分
    mdf_std_score = r_squared * std_score

    # 计算右偏标准分：beta * 修正标准分，进一步放大斜率较大时的得分
    rsk_std_score = beta * mdf_std_score

    return rsk_std_score