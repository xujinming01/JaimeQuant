import os
import glob
import sqlite3
import pandas as pd

def convert_csvs_to_sqlite(csv_dir='database', db_name='local_dayK.db'):
    """
    将指定目录下的所有 CSV 文件合并到一个 SQLite 数据库中。
    每次运行都会覆盖重建表，确保数据与 CSV 保持完全一致。
    """
    if not os.path.exists(csv_dir):
        print(f"错误: 找不到目录 '{csv_dir}'。请先运行爬虫脚本拉取数据。")
        return

    # 获取所有 csv 文件路径
    csv_files = glob.glob(os.path.join(csv_dir, '*.csv'))
    
    if not csv_files:
        print(f"在 '{csv_dir}' 目录下没有找到任何 CSV 文件。")
        return

    print(f"找到 {len(csv_files)} 个 CSV 文件，开始转换并构建 SQLite 数据库 '{db_name}'...")
    
    # 连接数据库 (如果不存在会自动创建)
    conn = sqlite3.connect(db_name)
    
    success_count = 0
    for file_path in csv_files:
        try:
            # 文件名通常为 513100.csv，提取出代码 513100 作为表名
            table_name = os.path.basename(file_path).replace('.csv', '')
            
            # 读取 CSV
            df = pd.read_csv(file_path)
            
            # 存入 SQLite，if_exists='replace' 意味着每次运行转换脚本都会用最新的CSV覆盖旧表
            df.to_sql(name=table_name, con=conn, if_exists='replace', index=False)
            
            # 【性能优化】为日期列建立索引。回测时按日期筛选数据的速度将提升几十倍
            if '日期' in df.columns:
                conn.execute(f'CREATE INDEX IF NOT EXISTS "idx_{table_name}_date" ON "{table_name}" ("日期")')
            
            print(f" [成功] {table_name}.csv -> 表 '{table_name}' ({len(df)} 行)")
            success_count += 1
            
        except Exception as e:
            print(f" [失败] 处理 {file_path} 时出错: {e}")
            
    # 提交事务并关闭连接
    conn.commit()
    conn.close()
    
    print("-" * 40)
    print(f"转换完成！成功转换 {success_count}/{len(csv_files)} 个文件。")
    print(f"你的回测数据库已就绪：{os.path.abspath(db_name)}")

if __name__ == "__main__":
    # 你可以修改这里的路径，默认转换 database 文件夹下的 CSV，输出为 dayK.db
    convert_csvs_to_sqlite(csv_dir='database', db_name='dayK.db')