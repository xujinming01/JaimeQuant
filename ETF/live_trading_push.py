import os
import requests

from live_trading import ETF_DICT, main

def send_wechat_push(title, content):
    """通过 PushPlus 发送微信推送"""
    token = os.environ.get("PUSHPLUS_TOKEN")
    if not token:
        print("⚠️ 未配置 PUSHPLUS_TOKEN 环境变量，跳过微信推送。")
        return
    
    url = "http://www.pushplus.plus/send"
    data = {
        "token": token,
        "title": title,
        "content": content,
        "template": "html"  # 使用 html 格式让换行更美观
    }
    try:
        response = requests.post(url, json=data)
        if response.status_code == 200:
            print("✅ 微信推送成功！")
        else:
            print(f"❌ 微信推送失败: {response.text}")
    except Exception as e:
        print(f"❌ 推送请求报错: {e}")

if __name__ == "__main__":
    prices, ranked, best_code, best_score = main()
    
    # 准备推送的内容文本（使用 HTML 表格和内联样式排版）
    # 在国内股市习惯中，红色代表上涨/正收益，绿色代表下跌/负收益
    push_content = f"""
    <h3 style="color: #333; margin-bottom: 10px;">📊 今日 ({prices.index[-1].strftime('%Y-%m-%d')}) 信号排名</h3>
    <table border="0" cellpadding="4" cellspacing="0" style="width: 100%; font-family: sans-serif; font-size: 14px; border-collapse: collapse;">
        <tr style="border-bottom: 2px solid #ccc; text-align: left; color: #666;">
            <th style="padding-bottom: 8px;">ETF名称</th>
            <th style="padding-bottom: 8px;">代码</th>
            <th style="text-align: right; padding-bottom: 8px;">动量得分</th>
        </tr>
    """
    
    for code, score in ranked.items():
        name_str = ETF_DICT[code]
        # 设置红涨绿跌颜色与带符号的格式化
        if score > 0:
            color = "#E60012" # 经典中国红
            score_str = f"+{score*100:.2f}%"
        elif score < 0:
            color = "#009944" # 经典护眼绿
            score_str = f"{score*100:.2f}%"
        else:
            color = "#333333" # 黑色平盘
            score_str = "0.00%"
            
        push_content += f"""
        <tr style="border-bottom: 1px solid #eee;">
            <td style="padding: 8px 0;">{name_str}</td>
            <td style="color: #888; padding: 8px 0;">{code}</td>
            <td style="text-align: right; color: {color}; font-weight: bold; padding: 8px 0;">{score_str}</td>
        </tr>
        """
        
    push_content += "</table>"
    
    # 结论部分
    push_content += """
    <div style="margin-top: 15px; padding: 10px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid #0056b3;">
        <h4 style="margin: 0 0 8px 0; color: #0056b3;">🎯 实盘建议操作</h4>
    """
    
    if best_score > 0:
        push_content += f"<p style='margin: 0; font-size: 15px;'>👉 满仓持有 / 买入 <br><strong style='color: #E60012; font-size: 18px;'>【{ETF_DICT[best_code]} ({best_code})】</strong></p></div>"
        push_title = f"量化调仓：买入 {ETF_DICT[best_code]}"
    else:
        push_content += "<p style='margin: 0; font-size: 14px; color: #E60012;'>⚠️ <strong>预警: 所有标的动量均为负数！</strong></p>"
        push_content += "<p style='margin: 5px 0 0 0; font-size: 15px;'>👉 <strong style='color: #009944;'>建议操作: 空仓，或买入避险资产 (如 161119)</strong></p></div>"
        push_title = "量化调仓：动量预警，建议空仓"
        
    # 终端依然保留纯文本输出（过滤掉 HTML 标签方便命令行查看）
    print(f"🎯 实盘建议操作: \n👉 {push_title.split('：')[1]}")
    
    # 触发微信推送
    send_wechat_push(push_title, push_content)