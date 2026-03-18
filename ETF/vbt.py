import os
import sqlite3
import numpy as np
import pandas as pd
import vectorbt as vbt
import quantstats as qs
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")


# 解决画图中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class VectorbtRotationStrategy:
    """
    基于 VectorBT 的 ETF 因子轮动回测框架
    支持多标的横向轮动、自定义因子、真实手续费与资金流转
    """
    def __init__(self, db_path, code_dict, safe_asset_code=None, benchmark_code='510300', start_date='20150101', end_date='20260101'):
        self.db_path = db_path
        self.code_dict = code_dict
        self.code_list = list(code_dict.keys())
        self.safe_asset_code = safe_asset_code
        self.benchmark_code = benchmark_code
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        
        # 存放原始数据的容器
        self.prices = None 
        
        # 初始化加载数据
        self._load_data()

    def _load_data(self):
        """从 SQLite 数据库提取数据并转为 vectorbt 最喜欢的宽表 (Wide DataFrame)"""
        print("📥 正在从本地数据库加载数据...")
        conn = sqlite3.connect(self.db_path)
        df_list = []
        
        # 确定需要加载的所有标的（风险资产池 + 单独指定的避险资产）
        all_codes = self.code_list.copy()
        if self.safe_asset_code and self.safe_asset_code not in all_codes:
            all_codes.append(self.safe_asset_code)
            
        for code in all_codes:
            try:
                query = f'SELECT 日期, 收盘 FROM "{code}"'
                df = pd.read_sql(query, conn)
                df['日期'] = pd.to_datetime(df['日期'])
                df = df[(df['日期'] >= self.start_date) & (df['日期'] <= self.end_date)]
                
                df = df.set_index('日期')
                df.rename(columns={'收盘': code}, inplace=True)
                df_list.append(df)
            except Exception as e:
                print(f"⚠️ 读取 {code} 失败: {e}")
                
        conn.close()
        
        # 合并所有标的，采用外连接保证日期对齐，前向填充缺失值
        self.prices = pd.concat(df_list, axis=1).sort_index().ffill()
        print(f"✅ 数据加载完成，形状: {self.prices.shape}")


    # ==================== 因子库 ====================
    
    def factor_pure_momentum(self, window=20):
        """简单因子：计算过去 N 天的动量涨幅"""
        print(f"🧮 计算因子: {window}日纯动量...")
        # 仅针对风险资产计算因子
        risk_prices = self.prices[self.code_list]
        factor_df = risk_prices / risk_prices.shift(window) - 1.0
        return factor_df

    def factor_atr_dynamic_score(self, lb_min=10, lb_max=60, vol_short_len=10, vol_long_len=60, ratio_cap=0.9):
        """复杂因子：复现你原先的 ATR 动态回溯窗口得分计算逻辑"""
        print(f"🧮 计算因子: ATR动态窗口回归得分...")
        
        # 仅针对风险资产计算因子
        risk_prices = self.prices[self.code_list]
        
        # 1. 计算日收益率和波动率矩阵
        returns = risk_prices.pct_change()
        vol_short = returns.rolling(vol_short_len).std()
        vol_long = returns.rolling(vol_long_len).std()
        
        # 2. 计算动态窗口大小
        vol_ratio_capped = (vol_short / vol_long).clip(upper=ratio_cap)
        lookback_df = np.floor(lb_min + (lb_max - lb_min) * (1 - vol_ratio_capped)).fillna(lb_max).astype(int)
        
        # 准备因子矩阵
        factor_df = pd.DataFrame(np.nan, index=risk_prices.index, columns=self.code_list)
        
        # 3. 纯 Numpy 加速的得分计算函数
        def fast_score(y):
            if len(y) < 2 or y[0] == 0: return np.nan
            y_norm = y / y[0]
            x = np.arange(1, len(y) + 1)
            # 最小二乘法斜率和R2
            cov = np.cov(x, y_norm, ddof=0)[0, 1]
            var_x = np.var(x, ddof=0)
            if var_x == 0: return np.nan
            slope = cov / var_x
            r_squared = np.corrcoef(x, y_norm)[0, 1] ** 2
            return 10000 * slope * r_squared
            
        # 4. 滚动计算得分
        for code in self.code_list:
            prices_arr = risk_prices[code].values
            lookback_arr = lookback_df[code].values
            scores = np.full(len(prices_arr), np.nan)
            
            for i in range(lb_max, len(prices_arr)):
                curr_lb = lookback_arr[i]
                window_prices = prices_arr[i - curr_lb + 1 : i + 1]
                scores[i] = fast_score(window_prices)
                
            factor_df[code] = scores
            
        return factor_df

    # ==================== 信号与执行 ====================

    def generate_target_weights(self, factor_df, top_n=1, enable_absolute_momentum=False, absolute_threshold=0.0):
        """
        根据因子值生成目标权重矩阵。
        
        参数:
        - factor_df: 因子值 DataFrame (现在里面只包含风险资产)
        - top_n: 每天选出因子值最大的前 n 只标的
        - enable_absolute_momentum: 绝对动量开关 (True/False)
        - absolute_threshold: 绝对动量阈值，仅当开关为 True 时生效
        """
        print(f"🎯 正在根据因子生成目标权重 (Top {top_n})...")
        
        # 1. 获取所有参与排名的风险资产
        risk_assets = list(factor_df.columns)
        
        # 2. 横向竞争（相对动量）：在风险资产中评出 Top N
        ranks = factor_df.rank(axis=1, ascending=False)
        mask_top = ranks <= top_n
        
        # 初始化风险资产的权重
        risk_weights = pd.DataFrame(0.0, index=factor_df.index, columns=risk_assets)
        risk_weights[mask_top] = 1.0 / top_n
        
        final_risk_weights = risk_weights.copy()
        
        # 3. 纵向过滤（绝对动量逻辑）
        if enable_absolute_momentum:
            print(f"🔒 绝对动量已开启 (阈值: {absolute_threshold})")
            # 得分必须大于阈值，否则没收权重变回 0.0
            pass_filter = factor_df > absolute_threshold
            final_risk_weights = risk_weights[pass_filter].fillna(0.0)
        else:
            print("🔓 绝对动量已关闭，仅使用相对排名分配权重")
            
        # 4. 计算剩余未分配的权重 (被没收的，或原来就不足的)
        remaining_weights = (1.0 - final_risk_weights.sum(axis=1)).round(4)
        
        # 5. 拼合最终的权重矩阵 (确保包含底仓价格矩阵中的所有列，包括避险资产)
        target_weights = pd.DataFrame(0.0, index=factor_df.index, columns=self.prices.columns)
        
        # 赋予风险资产权重
        target_weights[risk_assets] = final_risk_weights
        
        # 赋予避险资产权重
        if self.safe_asset_code and self.safe_asset_code in self.prices.columns:
            target_weights[self.safe_asset_code] = remaining_weights
        elif self.safe_asset_code:
            print(f"⚠️ 警告: 指定了避险资产 '{self.safe_asset_code}' 但数据读取失败，退化为【空仓】(持有现金)。")
        
        return target_weights

    def run_backtest(self, target_weights, init_cash=100000, fees=0.0001):
        """核心回测引擎"""
        print(f"🚀 开始撮合回测 (初始资金: {init_cash}, 费率: {fees})...")
        
        pf = vbt.Portfolio.from_orders(
            close=self.prices,
            size=target_weights,
            size_type='targetpercent',
            group_by=True,          
            cash_sharing=True,      
            init_cash=init_cash,
            fees=fees,
            slippage=0.000,         
            freq='D'                
        )
        print("✅ 回测完成！")
        return pf


# ================= 运行入口与分析 =================
if __name__ == "__main__":
    
    # 1. 配置参数 (请确保路径对齐你的项目结构)
    DB_PATH = 'code/database/dayK.db'
    
    # 💡纯净的字典：只存放参与横向打分排名的风险资产
    ETF_DICT = {
        '510880': '红利ETF华泰柏瑞',
        '159915': '创业板ETF易方达',
        '513100': '纳指ETF',
        '518880': '黄金ETF华安',
    }
    BENCHMARK_CODE = '513100'
    
    # 单独拿出来定义，不混杂在 ETF_DICT 中
    SAFE_ASSET_CODE = '161119' # 可以设为 None 来全局只空仓
    REPORT_TITLE = "ETF ATR"
    
    # 2. 实例化策略框架
    strategy = VectorbtRotationStrategy(
        db_path=DB_PATH,
        code_dict=ETF_DICT,
        safe_asset_code=SAFE_ASSET_CODE, # 👈 在初始化框架时，以专用通道传入避险资产
        benchmark_code=BENCHMARK_CODE,
        start_date='20140101',
        end_date='20260301'
    )
    
    factor_atr = strategy.factor_atr_dynamic_score()  # ATR 动态窗口得分因子 (现在底层严格只算 ETF_DICT 里的风险资产)
    
    # 3. 生成目标权重 
    weights_atr = strategy.generate_target_weights(
        factor_atr, 
        top_n=1,  
        # enable_absolute_momentum=True, 
        absolute_threshold=0.0         
    )
    
    pf_atr = strategy.run_backtest(weights_atr, init_cash=100000, fees=0.00006)

    # 4. 输出与可视化
    print("\n" + "="*40)
    print("📈 回测基础统计指标")
    print("="*40)
    print(pf_atr.stats())
    
    # 5. 结合 QuantStats 输出报告
    print("\n📝 正在生成 QuantStats HTML 报告...")
    strategy_returns = pf_atr.returns()
    benchmark_returns = strategy.prices[BENCHMARK_CODE].pct_change().fillna(0)
    strategy_returns, benchmark_returns = strategy_returns.align(benchmark_returns, join='inner')
    
    report_name = f"VectorBT_ATR_Rotation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    report_path = os.path.join(script_dir, report_name)
    
    qs.reports.html(
        returns=strategy_returns, 
        benchmark=benchmark_returns, 
        title=REPORT_TITLE, 
        output=report_path
    )
    print(f"✅ 报告已生成并保存至: {report_path}")

    # pf_atr.plot().show()