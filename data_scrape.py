import os
import time
import pandas as pd
import akshare as ak
from datetime import datetime, timedelta

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

class AkshareCsvBase:
    """
    通用数据爬取与入库基类 (CSV 版本)
    封装了CSV读写、增量日期计算逻辑
    """
    def __init__(self, csv_dir):
        self.csv_dir = csv_dir
        os.makedirs(self.csv_dir, exist_ok=True)
        
    def _get_start_date(self, code, default_start):
        """获取本地CSV中该代码的最新日期，计算增量拉取的起始日"""
        csv_path = os.path.join(self.csv_dir, f"{code}.csv")
        
        if os.path.exists(csv_path):
            try:
                # 只读取'日期'这一列来寻找最大值，节省内存和时间
                df_dates = pd.read_csv(csv_path, usecols=['日期'])
                max_date_val = df_dates['日期'].max()
                
                if pd.notna(max_date_val):
                    last_date = pd.to_datetime(max_date_val)
                    start_date = (last_date + timedelta(days=1)).strftime('%Y%m%d')
                    return start_date, last_date
            except Exception as e:
                print(f'[{code}] ⚠️ CSV查询异常或文件损坏: {e}')
        
        # 默认情况（文件不存在或读取失败）
        return default_start, None

    def _fetch_api_data(self, code, start_date, end_date):
        """
        [需子类实现] 调用具体的akshare接口获取数据
        """
        raise NotImplementedError("子类必须实现 _fetch_api_data 方法")

    def fetch_and_save(self, code_dict, default_start='19901219'):
        """
        核心调度流程：遍历字典 -> 检查本地日期 -> 调用API拉取 -> 追加存入CSV
        """
        end_date = datetime.now().strftime('%Y%m%d')
        
        for code, name in code_dict.items():
            start_date, last_date = self._get_start_date(code, default_start)
            
            if last_date:
                print(f'[{name} - {code}] 发现本地记录，最后更新至 {last_date.strftime("%Y-%m-%d")}，增量下载...')
            else:
                print(f'[{name} - {code}] 无本地记录，全量下载 ({start_date} 至今)...')
            
            if start_date > end_date:
                print(f'[{name} - {code}] 数据已是最新，跳过。\n')
                continue
                
            try:
                # 核心：调用子类具体实现的API抓取逻辑
                df = self._fetch_api_data(code, start_date, end_date)
                
                if df is None or df.empty:
                    print(f'[{name} - {code}] 该区间无新数据。\n')
                    continue
                
                # 通用数据处理：统一日期格式为 YYYY-MM-DD，让CSV更美观且标准
                if '日期' in df.columns:
                    df['日期'] = pd.to_datetime(df['日期']).dt.strftime('%Y-%m-%d')
                                
                # 追加进 CSV
                csv_path = os.path.join(self.csv_dir, f"{code}.csv")
                file_exists = os.path.exists(csv_path)
                
                # 如果文件不存在，写入表头(header=True)；存在则不写表头，直接追加(mode='a')
                df.to_csv(csv_path, mode='a', header=not file_exists, index=False, encoding='utf-8-sig')
                
                print(f'[{name} - {code}] 成功入库 {len(df)} 条数据。\n')
                time.sleep(1)  # 避免请求过快触发反爬机制
                
            except Exception as e:
                print(f'[{name} - {code}] ⚠️ 下载或保存失败: {e}\n')

    def get_raw_df(self, code):
        """读取单只标的完整数据"""
        csv_path = os.path.join(self.csv_dir, f"{code}.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df['日期'] = pd.to_datetime(df['日期'])
            return df.sort_values('日期').reset_index(drop=True)
        return pd.DataFrame()


# ================= 具体数据类 (继承基类) =================

class FundEtfHistEm(AkshareCsvBase):
    """ETF历史行情抓取类"""
    def _fetch_api_data(self, code, start_date, end_date):
        # 针对 ETF 开启复权 (adjust='hfq')
        return ak.fund_etf_hist_em(
            symbol=code, period='daily', 
            start_date=start_date, end_date=end_date, adjust='hfq'
        )
    
class FundLofHistEm(AkshareCsvBase):
    """LOF历史行情抓取类"""
    def _fetch_api_data(self, code, start_date, end_date):
        # 针对 LOF 开启复权 (adjust='hfq')
        return ak.fund_lof_hist_em(
            symbol=code, period='daily', 
            start_date=start_date, end_date=end_date, adjust='hfq'
        )

class IndexZhAHist(AkshareCsvBase):
    """A股指数历史行情抓取类"""
    def _fetch_api_data(self, code, start_date, end_date):
        # 针对 指数 的特定参数调用
        return ak.index_zh_a_hist(
            symbol=code, period="daily", 
            start_date=start_date, end_date=end_date
        )

class BondIndexHist(AkshareCsvBase):
    """债券指数历史行情抓取类"""
    def _fetch_api_data(self, code, start_date, end_date):
        # 债券接口不支持增量参数，此处直接拉取全量历史数据
        df = ak.bond_new_composite_index_cbond(indicator="财富", period="总值")
        
        if df is not None and not df.empty:
            # 将表头 'date' 统一重命名为 '日期'，'value' 重命名为 '收盘'
            df.rename(columns={'date': '日期', 'value': '收盘'}, inplace=True)
            
            # 为了配合基类的增量入库逻辑，避免重复插入，在本地手动过滤日期区间
            if '日期' in df.columns:
                # 转换为 datetime 对象以便安全比较
                df['日期'] = pd.to_datetime(df['日期'])
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                # 切片，只保留处于增量区间的数据
                df = df[(df['日期'] >= start_dt) & (df['日期'] <= end_dt)].copy()
                
        return df


# ================= 运行入口 =================
if __name__ == "__main__":
    # 确保目录存在
    db_path = 'database/'
    
    # 1. 定义需要抓取的字典
    etf_dict = {
        '513100': '纳指ETF',
        '518880': '黄金ETF华安',
        # '510300': '沪深300ETF华泰柏瑞',
        # '510500': '中证500ETF',
        # '510880': '红利ETF华泰柏瑞',
        '159915': '创业板ETF易方达',
        '159949': '创业板50ETF华安',
        '588000': '科创50ETF',
        '512890': '红利低波ETF华泰柏瑞',
        '512100': '中证1000ETF南方',
        '563300': '中证2000ETF华泰柏瑞',
        '159985': '豆粕ETF',
    }

    lof_dict = {
        '161119': '易方达新综债LOF',
    }
    
    index_dict = {
        '000016': '上证50',
        '000300': '沪深300',
    }

    bond_index_dict = {
        'CBA00101': '中债新综合财富',
    }

    # 2. 实例化对应的类并分配数据库文件
    etf_worker = FundEtfHistEm(db_path)
    lof_worker = FundLofHistEm(db_path)
    index_worker = IndexZhAHist(db_path)
    bond_index_worker = BondIndexHist(db_path)
    
    # 3. 执行更新操作
    print("====== 开始更新 ETF 数据 ======")
    etf_worker.fetch_and_save(etf_dict, default_start='19901219')
    
    print("====== 开始更新 LOF 数据 ======")
    lof_worker.fetch_and_save(lof_dict, default_start='19901219')
    
    print("====== 开始更新 指数 数据 ======")
    index_worker.fetch_and_save(index_dict, default_start='19901219')
    bond_index_worker.fetch_and_save(bond_index_dict, default_start='19901219')
    
    print("测试读取 上证50 指数 CSV：")
    try:
        df_50 = index_worker.get_raw_df('000016')
        print(df_50.tail())
    except Exception as e:
        print(f"读取测试失败: {e}")
