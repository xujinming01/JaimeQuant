from factors import calc_rsrs_advanced

"""filters.py
一组用于 ETF/资产筛选与风控的过滤器函数。

函数说明均使用中文文档字符串并返回与输入 `prices` / `highs` / `lows` 对应形状的
布尔掩码（True 表示通过过滤，可以被视为“安全”或“可选”）。
"""


def filter_recent_drop(prices, max_drop_pct: float = 0.05):
    """过滤近期出现大幅下跌或连续下跌的资产。

    规则：
    - con1: 近 3 个交易日内任意 1 日跌幅超过 `max_drop_pct`。
    - con2: 连续 3 日下跌，且 3 日累计跌幅超过 `max_drop_pct`。
    - con3: 昨天触发了 con2（即对连续下跌存在滞后判定）。

    参数
    - prices: pandas.Series 或 pandas.DataFrame，按时间索引的价格序列。
    - max_drop_pct: 允许的最大跌幅（例如 0.05 表示 5%）。

    返回
    - 布尔掩码（同 shape）：True 表示安全（通过过滤）；False 表示被过滤掉。
    """
    drop_threshold = 1.0 - max_drop_pct
    print(f"🛡️ 计算过滤器: 近期下跌过滤 (最大允许跌幅: {max_drop_pct*100}%)")

    # 计算日收益率（今天 / 昨天）
    daily_ret = prices / prices.shift(1)

    # con1：近 3 天内存在任意一天的跌幅超过阈值
    is_large_drop = daily_ret < drop_threshold
    con1 = is_large_drop.rolling(window=3).max().fillna(0).astype(bool)

    # con2：连续 3 天下跌，且 3 天累计跌幅超过阈值
    is_down = prices < prices.shift(1)
    three_down = is_down & is_down.shift(1) & is_down.shift(2)
    three_day_drop = (prices / prices.shift(3)) < drop_threshold
    con2 = three_down & three_day_drop

    # con3：昨日触发了 con2（对连续下跌的滞后判定）
    con3 = con2.shift(1).fillna(False)

    is_danger = con1 | con2 | con3
    # 返回“通过”掩码：非危险即为通过
    return ~is_danger


def filter_rsrs_timing(highs, lows, window: int = 16, z_window: int = 300, threshold: float = -0.7):
    """基于 RSRS（回归斜率标准化）右偏 z-score 的择时过滤器。

    依据（常用研报标准）：
    - RSRS z-score > 0.7：偏多
    - RSRS z-score < -0.7：偏空
    - 介于两者之间：观望

    参数
    - highs: 最高价序列（Series 或 DataFrame）。
    - lows: 最低价序列（Series 或 DataFrame）。
    - window: 用于斜率回归的回溯窗口（默认 16）。
    - z_window: 用于计算 z-score 的滚动窗口长度（默认 300）。
    - threshold: 判定为安全的阈值（默认 -0.7，可按策略调整）。

    返回
    - 布尔掩码：True 表示 RSRS 值 >= threshold（被视为安全/通过）。
    """
    print(f"🛡️ 计算过滤器: RSRS 高级右偏标准分 (阈值: {threshold})")
    rsrs_matrix = calc_rsrs_advanced(highs, lows, window=window, z_window=z_window)
    is_safe_mask = rsrs_matrix >= threshold
    return is_safe_mask


def filter_absolute_momentum(prices, window: int = 120, threshold: float = 0.0):
    """绝对动量过滤器：比较当前价格与 `window` 日前价格的百分比变化。

    参数
    - prices: pandas.Series 或 pandas.DataFrame，按时间索引的价格序列。
    - window: 用于计算动量的时间窗口（单位：交易日），例如 120 表示 120 日动量。
    - threshold: 动量阈值，默认 0.0（要求期末价格高于期初价格）。

    返回
    - 布尔掩码：True 表示动量 > threshold（通过）。
    """
    print(f"🛡️ 计算过滤器: {window}日 绝对动量过滤 (得分阈值: {threshold})")
    momentum = (prices / prices.shift(window)) - 1.0
    return momentum > threshold