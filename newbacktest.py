import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import math
# 已考虑手续费，默认手续费开双边*2
#没有金融期货的数据（IC IF IM, T , TF) 后面有了可以补上，回测的话信号要删掉这些，WR没数据，没什么成交量
# 该回测函数仅针对期货交易

import pandas as pd

contract = pd.read_csv('C:\\Users\\Administrator\\Desktop\\MLBasedFuturesStrategy\\MLBasedFuturesStrategy\\combined_data.csv', encoding='utf-8')


def get_future_data(df, date, code):
    # 读取数据，使用GBK编码

    # 将日期列转换为datetime类型，以确保可以进行日期比较
    df['time'] = pd.to_datetime(df['time'])
    # 替换'--'为NaN
    df.replace('--', np.nan, inplace=True)

    # 将输入的日期字符串也转换为datetime类型
    input_date = pd.to_datetime(date)

    # 筛选符合输入日期和代码的行
    filtered_data = df[(df['time'] == input_date) & (df['ths_future_code_future_processed'] == code)]

    # 检查筛选结果是否为空
    if filtered_data.empty:
        return "No data available for the given date and code."

    # 获取所需的列信息，如果'rate'是null，则使用'fee'
    result = filtered_data.apply(lambda row: {
        'code': row['ths_future_code_future_processed'],
        'contract_multiplier': int(row['ths_contract_multiplier']),
        'deposit': float(row['ths_trade_deposit_future']),
        'transactionfee': float(row['ths_transaction_procedure_rate_future'] if (pd.notnull(row['ths_transaction_procedure_rate_future']) and row['ths_transaction_procedure_rate_future'] != 0)else row['ths_transaction_procedure_fee_future']),

        'opentime': row['ths_open_time_night_future'] if pd.notnull(row['ths_open_time_night_future']) else row[
            'ths_open_time_day_future'],

    }, axis=1)

    # 返回结果
    return result.iloc[0]  # 假设每个日期和代码的组合只有一行数据





def calculate_trade_return(row):
    # 计算基础收益，计算方法为手数 * （收盘价- 开盘价） * 点数
    # 注意需要读取未进行换月连续化的数据进行计算，以提高准确性
    
    row['Return'] = row['Direction'] * (row['Close_Price'] - row['Open_Price']) * (get_future_data(contract,row['Open_Time'],row['Asset'])['contract_multiplier'])
    
    # 判断手续费是固定还是按比例,默认手续费开双边 *2
    if get_future_data(contract,row['Open_Time'],row['Asset'])['transactionfee'] < 1:
        # 如果手续费按比例收，计算方法为（开价+收价）* 点数 * 手数绝对值 * 比例
        commission = ((get_future_data(contract,row['Open_Time'],row['Asset'])['transactionfee'] 
                      * get_future_data(contract,row['Open_Time'],row['Asset'])['contract_multiplier'] * (row['Open_Price'] + row['Close_Price']))
                      * abs(row['Direction'])*2)

    else:  # 固定手续费，直接用手续费值 * 手数绝对值 * 2即可
        commission = get_future_data(contract,row['Open_Time'],row['Asset'])['transactionfee'] * 2 * abs(row['Direction'])
    # 最终收益 = 基础收益 - 手续费
    row['Return'] = row['Return'] - commission
    return row



def backtest(initial_capital, trade_records):
    # 该函数用于回测，输入为初始资金和交易记录
    # 整理交易记录的格式
    df = pd.DataFrame(trade_records,
                      columns=['Asset', 'Open_Time', 'Open_Price', 'Direction', 'Close_Time', 'Close_Price'])

    # 将平仓时间的格式转化为时间戳
    df['Close_Time'] = pd.to_datetime(df['Close_Time'])

    # 按照平仓时间排序， 以平仓时间作为一次交易结束并结算的时间
    df = df.sort_values(by='Close_Time')

    # 计算每笔交易的收益（转化单位为人民币，考虑手续费）
    df = df.apply(calculate_trade_return, axis=1)

    # 计算资金曲线
    # 首先设置日期为索引并对其进行重采样
    df = df.set_index('Close_Time')
    daily_returns = df.resample('D').sum(numeric_only=True).fillna(0)['Return']
    cumulative_net_value = (daily_returns.cumsum() + initial_capital) / initial_capital
    daliy_rerutns_percent = cumulative_net_value.pct_change()

    # 计算最大资金序列
    peak = cumulative_net_value.cummax()

    # 计算回撤
    drawdown = (peak - cumulative_net_value) / peak

    # 计算最大回撤
    drawdown_max = drawdown.max()

    # 计算最大回撤的结束日期
    end_date = drawdown.idxmax()

    # 为了找到最大回撤的开始日期，我们要找从开始到结束日期中资金达到其最高点的日期
    start_date = cumulative_net_value.loc[:end_date].idxmax()

    print("最大回撤开始日期:", start_date)
    print("最大回撤结束日期:", end_date)

    # 定义无风险利率，此处假设为0.02（或2%）
    risk_free_rate = 0.02

    # 计算总收益率
    total_return = cumulative_net_value.iloc[-1] - 1

    # 计算年化收益率
    days = (cumulative_net_value.index[-1] - cumulative_net_value.index[0]).days
    annualized_return = (cumulative_net_value.iloc[-1]) ** (365.25 / days) - 1

    # 计算年化波动率
    annualized_volatility = daliy_rerutns_percent.std() * np.sqrt(252)

    # 计算夏普比率
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # 统计交易次数
    total_trades = len(df)

    # 统计交易胜率
    win_rate = len(df[df['Return'] > 0]) / total_trades

    # 统计日胜率
    daily_win_rate = len(daily_returns.loc[daily_returns > 0]) / len(daily_returns.loc[daily_returns != 0])

    # 计算平均单笔盈亏
    average_returns = df['Return'].sum() / len(df)

    # 计算盈亏比
    profit_average = df.loc[df['Return'] > 0]['Return'].sum() / len(df.loc[df['Return'] > 0])
    lose_average = df.loc[df['Return'] < 0]['Return'].sum() / len(df.loc[df['Return'] < 0])
    profit_loss_ratio = profit_average / abs(lose_average)

    # 计算日盈亏比
    daily_profit_average = (
            daily_returns.loc[daily_returns > 0].sum() /
            len(daily_returns.loc[daily_returns > 0])
    )
    daily_loss_average = (
            daily_returns.loc[daily_returns < 0].sum() /
            len(daily_returns.loc[daily_returns < 0])
    )
    daliy_profit_loss_ratio = daily_profit_average / abs(daily_loss_average)

    # 计算卡玛比率
    calmar_ratio = annualized_return / drawdown_max

    # 计算百次交易盈亏
    hundred_trades_profit_loss = win_rate * (profit_loss_ratio + 1) - 1

    # 绘制资产净值图

    print(f"交易次数: {total_trades}, "
          f"胜率: {win_rate:.2%}, "
          f"日胜率: {daily_win_rate:.2%}, "
          f"盈亏比: {profit_loss_ratio:.3f}, "
          f"日盈亏比: {daliy_profit_loss_ratio:.3f}, "
          f"单笔平均盈亏: {average_returns:.3f}, "
          f"最大回撤: {drawdown_max:.3f}, "
          f"总收益率: {total_return:.2%}, "
          f"年化收益率: {annualized_return:.2%}, "
          f"夏普比率: {sharpe_ratio:.3f}, "
          f"卡玛比率: {calmar_ratio:.3f}, "
          f"百次交易盈亏{hundred_trades_profit_loss}")
    plt.figure(figsize=(10, 6))
    cumulative_net_value.plot(title='net worth')
    plt.xlabel("date")
    plt.ylabel("net worth ratio")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return {
        '交易次数': total_trades,
        '胜率': win_rate,
        '日胜率': daily_win_rate,
        '盈亏比': profit_loss_ratio,
        '日盈亏比': daliy_profit_loss_ratio,
        '单笔平均盈亏': average_returns,
        '最大回撤': drawdown_max,
        # 'Max Drawdown Date': start_date,
        '总收益率': total_return,
        '年化收益率': annualized_return,
        '夏普比率': sharpe_ratio,
        '卡玛比率': calmar_ratio,
        '百次交易盈亏': hundred_trades_profit_loss
    }, df, daily_returns


def analyze_trade_records(trade_records):
    # 该函数用于统计分析一份交易记录中各个期货品种的表现情况，最终返回一个存储了各个品种的开始交易时间、结束交易时间、交易次数、胜率、收益的dataframe

    # 计算每笔交易的收益（转化单位为人民币，考虑手续费）
    trade_records = trade_records.apply(calculate_trade_return, axis=1)

    # 通过id分组并计算
    grouped = trade_records.groupby('Asset')
    summery = pd.DataFrame()
    summery['Open_Time'] = grouped['Open_Time'].min()
    summery['Close_Time'] = grouped['Close_Time'].max()
    summery['trades'] = grouped.size()
    summery['win_rate'] = grouped.apply(lambda x: (x['Return'] >= 0).sum() / len(x))
    summery['profit'] = grouped.apply(lambda x: x['Return'].sum())

    # 重置索引
    summery.reset_index(inplace=True)

    return summery


def equal_weight(data, market_value):
    # 该函数用于将交易记录中的手数根据等市值原则进行统一,输入为一个交易记录和规定的市值大小
    def hands_calculate(row):
        commission_value = row['Open_Price'] * get_future_data(contract,row['Open_Time'],row['Asset'])['contract_multiplier']
        if commission_value >= market_value:
            row['Direction'] = row['Direction']  # 可以在这里添加你的逻辑
        else:
            row['Direction'] = row['Direction'] * int(market_value / commission_value)
        row['Position'] = abs(row['Direction']) * get_future_data(contract,row['Open_Time'],row['Asset'])['deposit']
        return row

    # 在这里设置 axis=1 以对行应用函数
    data = data.apply(hands_calculate, axis=1)
    return data

def equal_weight_l(data, market_value, bar):
    # 该函数用于将交易记录中的手数根据等市值原则进行统一,输入为一个交易记录和规定的市值大小
    def hands_calculate(row):
        commission_value = row['Open_Price'] * get_future_data(contract,row['Open_Time'],row['Asset'])['contract_multiplier']
        if commission_value >= market_value:
            row['Direction'] = row['Direction']  # 可以在这里添加你的逻辑
        elif abs(row['Predicted_Return']) > 2 * bar:
            row['Direction'] = row['Direction'] * int(market_value / commission_value) * 2
        else:
            row['Direction'] = row['Direction'] * int(market_value / commission_value)
        row['Position'] = abs(row['Direction']) * get_future_data(contract,row['Open_Time'],row['Asset'])['deposit']
        return row

    # 在这里设置 axis=1 以对行应用函数
    data = data.apply(hands_calculate, axis=1)
    return data


def prediciton_weight(data, market_value):
    def hands_calculate(row):
        commission_value = row['Open_Price'] * get_future_data(contract,row['Open_Time'],row['Asset'])['contract_multiplier']
        if commission_value >= market_value:
            row['Direction'] = row['Direction']  # 可以在这里添加你的逻辑
        else:
            row['Direction'] = row['Direction'] * int(market_value / commission_value)
        return row
    
    def omit_drawback(row):
        commission_rate = get_future_data(contract,row['Open_Time'],row['Asset'])['contract_multiplier']
        if commission_rate >= abs(row['Predicted_Return']):
            row['Direction'] = 0
        return row
    
    def weighted_hands(row):
        weight = np.exp(row['Predicted_Return']) / sum(np.exp(data['Predicted_Return']))
        row['Direction'] = row['Direction'] * weight
        return row
    data = data.apply(omit_drawback, axis = 1)
    data = data.apply(hands_calculate, axis = 1)
    data = data.apply(weighted_hands, axis = 1)

    return data
