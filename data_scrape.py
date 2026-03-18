import os
import time
import sqlite3
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

class AkshareDbBase:
    """
    通用数据爬取与入库基类
    封装了SQLite连接、增量日期计算、通用入库逻辑
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path)
        
    def _get_start_date(self, code, default_start):
        """获取本地数据库中该代码的最新日期，计算增量拉取的起始日"""
        try:
            query = f'SELECT MAX(日期) FROM "{code}"'
            max_date_val = pd.read_sql(query, self.conn).iloc[0, 0]
            
            if pd.notna(max_date_val):
                if isinstance(max_date_val, str):
                    last_date = datetime.strptime(max_date_val[:10], '%Y-%m-%d')
                else:
                    last_date = max_date_val
                    
                start_date = (last_date + timedelta(days=1)).strftime('%Y%m%d')
                return start_date, last_date
            
        except (sqlite3.OperationalError, pd.errors.DatabaseError) as e:
            if "no such table" not in str(e).lower():
                print(f'[{code}] ⚠️ 数据库查询异常: {e}')
        
        # 默认情况（表不存在或为空）
        return default_start, None

    def _fetch_api_data(self, code, start_date, end_date):
        """
        [需子类实现] 调用具体的akshare接口获取数据
        """
        raise NotImplementedError("子类必须实现 _fetch_api_data 方法")

    def fetch_and_save(self, code_dict, default_start='19901219'):
        """
        核心调度流程：遍历字典 -> 检查本地日期 -> 调用API拉取 -> 格式化存入DB
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
                
                # 通用数据处理
                if '日期' in df.columns:
                    df['日期'] = pd.to_datetime(df['日期'])
                                
                # 追加进数据库
                df.to_sql(name=code, con=self.conn, if_exists='append', index=False)
                print(f'[{name} - {code}] 成功入库 {len(df)} 条数据。\n')
                time.sleep(3)
                
            except Exception as e:
                print(f'[{name} - {code}] ⚠️ 下载或保存失败: {e}\n')

    def get_raw_df(self, code):
        """读取单只标的完整数据"""
        query = f'SELECT * FROM "{code}" ORDER BY 日期'
        return pd.read_sql(query, self.conn)

    def close(self):
        self.conn.close()


# ================= 具体数据类 (继承基类) =================

class FundEtfHistEm(AkshareDbBase):
    """ETF历史行情抓取类"""
    def _fetch_api_data(self, code, start_date, end_date):
        # 针对 ETF 开启复权 (adjust='hfq')
        return ak.fund_etf_hist_em(
            symbol=code, period='daily', 
            start_date=start_date, end_date=end_date, adjust='hfq'
        )
    
class FundLofHistEm(AkshareDbBase):
    """LOF历史行情抓取类"""
    def _fetch_api_data(self, code, start_date, end_date):
        # 针对 LOF 开启复权 (adjust='hfq')
        return ak.fund_lof_hist_em(
            symbol=code, period='daily', 
            start_date=start_date, end_date=end_date, adjust='hfq'
        )

class IndexZhAHist(AkshareDbBase):
    """A股指数历史行情抓取类"""
    def _fetch_api_data(self, code, start_date, end_date):
        # 针对 指数 的特定参数调用
        return ak.index_zh_a_hist(
            symbol=code, period="daily", 
            start_date=start_date, end_date=end_date
        )

class BondIndexHist(AkshareDbBase):
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
    db_path = 'code/database/'
    os.makedirs(db_path, exist_ok=True)
    
    # 1. 定义需要抓取的字典
    etf_dict = {
        '510300': '沪深300ETF华泰柏瑞',
        '510500': '中证500ETF',
        '510880': '红利ETF华泰柏瑞',
        '159915': '创业板ETF易方达',
        '513100': '纳指ETF',
        '518880': '黄金ETF华安',
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
    # 可以选择存在同一个db文件里，也可以分开存。这里演示分开存。
    etf_db = FundEtfHistEm(os.path.join(db_path, 'dayK.db'))
    lof_db = FundLofHistEm(os.path.join(db_path, 'dayK.db'))
    index_db = IndexZhAHist(os.path.join(db_path, 'dayK.db'))
    bond_index_db = BondIndexHist(os.path.join(db_path, 'dayK.db'))
    # 3. 执行更新操作
    print("====== 开始更新 ETF 数据 ======")
    etf_db.fetch_and_save(etf_dict, default_start='19901219')
    
    print("====== 开始更新 LOF 数据 ======")
    lof_db.fetch_and_save(lof_dict, default_start='19901219')
    
    print("====== 开始更新 指数 数据 ======")
    index_db.fetch_and_save(index_dict, default_start='19901219')
    bond_index_db.fetch_and_save(bond_index_dict, default_start='19901219')
    
    # 测试读取
    print("测试读取 上证50 指数：")
    try:
        df_50 = index_db.get_raw_df('000016')
        print(df_50.tail())
    except Exception as e:
        print(f"读取测试失败: {e}")
    
    # 4. 关闭连接
    etf_db.close()
    lof_db.close()
    index_db.close()
    bond_index_db.close()