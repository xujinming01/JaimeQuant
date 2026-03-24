import numpy as np
import pandas as pd


def generate_donchian_rotation_weights(strategy, window=20):
    """
    海龟通道 + 动量轮动 状态机生成目标权重
    """
    print(f"🐢 正在生成 海龟+动量轮动 目标权重 (窗口: {window}日)...")
    
    # 提取风险资产数据
    risk_codes = strategy.code_list
    prices = strategy.prices[risk_codes]
    highs = strategy.highs[risk_codes]
    lows = strategy.lows[risk_codes]
    
    # 1. 计算轮动排序：现价相对于N日前收盘价的涨幅
    momentum = prices / prices.shift(window) - 1.0
    
    # 使用 method='first' 保证每天即使有涨幅相同的，也只选出唯一的一个第一名
    ranks = momentum.rank(axis=1, ascending=False, method='first')
    is_rank1 = (ranks == 1) # 布尔型 DataFrame，标记每天的第一名
    
    # 2. 计算唐奇安通道（20日最高、最低点），shift(1) 避免未来函数
    high_20 = highs.shift(1).rolling(window).max()
    low_20 = lows.shift(1).rolling(window).min()
    
    # 3. 构建核心信号矩阵 (布尔型)
    # 【开仓条件】：必须是当天的第一名，且现价突破前20日最高点
    entry_signal = is_rank1 & (prices > high_20)
    
    # 【平仓条件】：不再是第一名（排名变更），或者 现价跌破前20日最低点
    exit_signal = (~is_rank1) | (prices < low_20)
    
    # 4. 向量化状态机核心：利用 NaN 和 ffill (前向填充) 传播持仓状态
    signals = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    
    # 赋值信号（注：因为平仓包含~is_rank1，开仓包含is_rank1，所以同一天绝对不会发生冲突）
    signals[entry_signal] = 1.0
    signals[exit_signal] = 0.0
    
    # 向前填充状态（1会一直延续到遇到0），最初始的 NaN 填 0
    target_weights = signals.ffill().fillna(0.0)
    
    # 5. 处理避险资产分配
    if strategy.safe_asset_code:
        # 每行风险资产的权重总和 (因为限定了 rank1，最大和永远是 1)
        risk_exposure = target_weights.sum(axis=1)
        # 空仓部分自动买入国债/避险资产
        target_weights[strategy.safe_asset_code] = 1.0 - risk_exposure
        
    return target_weights

