import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
import math
# 已考虑手续费，考虑到滑点现象，手续费按照1.5倍计算。
# 该回测函数仅针对期货交易


def futures_trading_data():
    # 创建一个空字典来存储期货品种和手续费、保证金、点数算法的映射关系
    final_data = {}

    # 期货品种和对应手续费算法的列表
    # 因为双开的期货品种远多于单开的期货品种，所以这里默认所有期货品种的手续费都是双开，单开的品种手续费除以二
    # 后续写作回测程序时用双开来算，但由于单开的品种手续费已被除2，最终回测效果相同。
    # 另外，有些品种手续费按比例收，有些品种手续费固定收，再此不做区分，只需在回测函数中加入判断即可
    # 数量级很小的即为按比例收取

    data = [
        ['代码', '手续费', '保证金', '点数', '交易所', '开盘时间', '最小价格变动单位'],
        ['IF', 0.00003, 0.12, 300, 'CFFEX', '9:30:00', 0.2],
        ['IC', 0.00003, 0.12, 200, 'CFFEX', '9:30:00', 0.2],
        ['IM', 0.00003, 0.12, 200, 'CFFEX', '9:30:00', 0.2],
        ['IH', 0.00003, 0.12, 300, 'CFFEX', '9:30:00', 0.2],
        ['T', 4.5 / 2, 0.02, 10000, 'CFFEX', '9:00:00', 0.005],
        ['TF', 4.5 / 2, 0.012, 10000, 'CFFEX', '9:00:00', 0.005],
        ['TS', 4.5 / 2, 0.005, 10000, 'CFFEX', '9:00:00', 0.002],
        ['TA', 4.5 / 2, 0.07, 5, 'CZCE', '21:00:00', 2],
        ['SR', 4.5 / 2, 0.07, 10, 'CZCE', '21:00:00', 1],
        ['FG', 9, 0.12, 20, 'CZCE', '21:00:00', 1],
        ['PF', 4.5, 0.08, 5, 'CZCE', '21:00:00', 2],
        ['SA', 0.0005, 0.12, 20, 'CZCE', '21:00:00', 1],
        ['MA', 0.0002, 0.1, 10, 'CZCE', '21:00:00', 1],
        ['CF', 6 / 2, 0.07, 5, 'CZCE', '21:00:00', 5],
        ['CY', 6 / 2, 0.1, 5, 'CZCE', '21:00:00', 5],
        ['AP', 8, 0.1, 10, 'CZCE', '9:00:00', 1],
        ['CJ', 15, 0.12, 5, 'CZCE', '9:00:00', 5],
        ['RM', 2, 0.09, 10, 'CZCE', '21:00:00', 1],
        ['RS', 3, 0.2, 10, 'CZCE', '9:00:00', 1],
        ['OI', 3, 0.09, 10, 'CZCE', '21:00:00', 1],
        ['PK', 6, 0.08, 5, 'CZCE', '9:00:00', 2],
        ['SF', 4.5 / 2, 0.12, 5, 'CZCE', '9:00:00', 2],
        ['SM', 4.5 / 2, 0.12, 5, 'CZCE', '9:00:00', 2],
        ['UR', 7, 0.08, 20, 'CZCE', '9:00:00', 1],
        ['ZC', 155, 0.5, 100, 'CZCE', '21:00:00', 0.2],
        ['WH', 450, 0.15, 20, 'CZCE', '9:00:00', 1],
        ['PM', 45, 0.15, 50, 'CZCE', '9:00:00', 1],
        ['RI', 4.5, 0.15, 20, 'CZCE', '9:00:00', 1],
        ['LR', 4.5, 0.15, 20, 'CZCE', '9:00:00', 1],
        ['JR', 4.5, 0.15, 20, 'CZCE', '9:00:00', 1],
        ['I', 0.00022, 0.13, 100, 'DCE', '21:00:00', 0.5],
        ['J', 0.00016, 0.2, 100, 'DCE', '21:00:00', 0.5],
        ['JM', 0.00033, 0.2, 60, 'DCE', '21:00:00', 0.5],
        ['M', 2.3, 0.07, 10, 'DCE', '21:00:00', 1],
        ['Y', 3.6, 0.07, 10, 'DCE', '21:00:00', 2],
        ['P', 3.6, 0.08, 10, 'DCE', '21:00:00', 2],
        ['A', 3, 0.08, 10, 'DCE', '21:00:00', 1],
        ['B', 1.5, 0.08, 10, 'DCE', '21:00:00', 1],
        ['BB', 0.00011, 0.4, 500, 'DCE', '21:00:00', 0.05],
        ['FB', 0.00011, 0.1, 10, 'DCE', '9:00:00', 0.5],
        ['C', 1.8, 0.08, 10, 'DCE', '21:00:00', 1],
        ['CS', 2.2, 0.06, 10, 'DCE', '21:00:00', 1],
        ['RR', 1.5, 0.06, 10, 'DCE', '21:00:00', 1],
        ['L', 1.5, 0.07, 5, 'DCE', '21:00:00', 1],
        ['V', 1.5, 0.07, 5, 'DCE', '21:00:00', 1],
        ['PP', 1.5, 0.07, 5, 'DCE', '21:00:00', 1],
        ['EB', 4.5, 0.08, 5, 'DCE', '21:00:00', 1],
        ['EG', 4.5, 0.08, 10, 'DCE', '21:00:00', 1],
        ['JD', 0.00017, 0.08, 5, 'DCE', '9:00:00', 1],
        ['LH', 0.00011, 0.12, 16, 'DCE', '9:00:00', 5],
        ['PG', 9, 0.08, 20, 'DCE', '21:00:00', 1],
        ['SC', 30 / 2, 0.1, 1000, 'INE', '21:00:00', 0.1],
        ['LU', 0.0002, 0.1, 10, 'INE', '21:00:00', 1],
        ['NR', 0.0003 / 2, 0.08, 10, 'INE', '21:00:00', 5],
        ['BC', 0.0002 / 2, 0.08, 5, 'INE', '21:00:00', 10],
        ['CU', 0.0008, 0.1, 5, 'SHFE', '21:00:00', 10],
        ['AL', 4.5, 0.1, 5, 'SHFE', '21:00:00', 5],
        ['AO', 0.0002 / 2, 0.09, 20, 'SHFE', '21:00:00', 1],
        ['ZN', 4.5 / 2, 0.1, 5, 'SHFE', '21:00:00', 5],
        ['PB', 0.0006 / 2, 0.1, 5, 'SHFE', '21:00:00', 5],
        ['NI', 4.5, 0.12, 1, 'SHFE', '21:00:00', 10],
        ['SN', 4.5, 0.12, 1, 'SHFE', '21:00:00', 10],
        ['AG', 0.0002, 0.09, 15, 'SHFE', '21:00:00', 1],
        ['AU', 3.5 / 2, 0.08, 1000, 'SHFE', '21:00:00', 0.02],
        ['RB', 0.00011, 0.07, 10, 'SHFE', '21:00:00', 1],
        ['HC', 0.00011, 0.07, 10, 'SHFE', '21:00:00', 1],
        ['SS', 3.5 / 2, 0.07, 5, 'SHFE', '21:00:00', 5],
        ['FU', 0.0008 / 2, 0.1, 10, 'SHFE', '21:00:00', 1],
        ['BU', 0.00011, 0.1, 10, 'SHFE', '21:00:00', 1],
        ['WR', 0.0006 / 2, 0.09, 10, 'SHFE', '9:00:00', 1],
        ['RU', 4.5 / 2, 0.08, 10, 'SHFE', '21:00:00', 5],
        ['SP', 0.0007 / 2, 0.08, 10, 'SHFE', '21:00:00', 2],
        ['SI', 0.00011 / 2, 0.12, 5, 'GFEX', '9:00:00', 5],
    ]

    # 遍历数据列表，将品种和手续费、保证金比例、点数（实际人民币（单位元）变化：数据变化）添加到字典中
    for item in data:
        symbol = item[0]  # 品种名称
        algorithm = tuple(item[1:7])  # 手续费、保证金比例、点数、交易所名称、开盘时间、最小价格变动单位
        final_data[symbol] = algorithm
    return final_data


commission_info = futures_trading_data()


def calculate_trade_return(row):
    # 计算基础收益，计算方法为手数 * （收盘价- 开盘价） * 点数
    # 注意需要读取未进行换月连续化的数据进行计算，以提高准确性
    row['Return'] = row['Direction'] * (row['Close_Price'] - row['Open_Price']) * commission_info[row['Asset']][2]

    # 判断手续费是固定还是按比例
    if commission_info[row['Asset']][0] < 1:
        # 如果手续费按比例收，计算方法为（开价+收价）* 点数 * 手数绝对值 * 比例
        commission = ((commission_info[row['Asset']][0]
                      * commission_info[row['Asset']][2] * (row['Open_Price'] + row['Close_Price']))
                      * abs(row['Direction']))

    else:  # 固定手续费，直接用手续费值 * 手数绝对值 * 2即可
        commission = commission_info[row['Asset']][0] * 2 * abs(row['Direction'])
    # 最终收益 = 基础收益 - 手续费
    row['Return'] = row['Return'] - commission
    return row


# def day_analyze(trade_records):

#     # 创建pandas DataFrame
#     df = pd.DataFrame(trade_records,
#                       columns=['Asset', 'Open_Time', 'Open_Price', 'Direction', 'Close_Time', 'Close_Price'])

#     df['Close_Time'] = pd.to_datetime(df['Close_Time'])

#     # 按照平仓时间排序
#     df = df.sort_values(by='Close_Time')

#     # 计算每笔交易的收益（转化单位为人民币，考虑手续费）
#     df = df.apply(calculate_trade_return, axis=1)
#     # 计算资金曲线

#     # 设置日期为索引并对其进行重采样
#     df = df.set_index('Close_Time')
#     daily_returns = df.resample('D').sum(numeric_only=True).fillna(0)['Return']
#     return daily_returns


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
        commission_value = row['Open_Price'] * commission_info[row['Asset']][2]
        if commission_value >= market_value:
            row['Direction'] = row['Direction']  # 可以在这里添加你的逻辑
        else:
            row['Direction'] = row['Direction'] * int(market_value / commission_value)
        row['Position'] = abs(row['Direction']) * commission_info[row['Asset']][1]*commission_info[row['Asset']][2] * row['Open_Price']
        return row

    # 在这里设置 axis=1 以对行应用函数
    data = data.apply(hands_calculate, axis=1)
    return data

def equal_weight_l(data, market_value, bar):
    # 该函数用于将交易记录中的手数根据等市值原则进行统一,输入为一个交易记录和规定的市值大小
    def hands_calculate(row):
        commission_value = row['Open_Price'] * commission_info[row['Asset']][2]
        if commission_value >= market_value:
            row['Direction'] = row['Direction']  # 可以在这里添加你的逻辑
        elif abs(row['Predicted_Return']) > 3 * bar:
            row['Direction'] = row['Direction'] * int(market_value / commission_value) * 2
        else:
            row['Direction'] = row['Direction'] * int(market_value / commission_value)
        row['Position'] = abs(row['Direction']) * commission_info[row['Asset']][1]*commission_info[row['Asset']][2] * row['Open_Price']
        return row

    # 在这里设置 axis=1 以对行应用函数
    data = data.apply(hands_calculate, axis=1)
    return data


def prediciton_weight(data, market_value):
    def hands_calculate(row):
        commission_value = row['Open_Price'] * commission_info[row['Asset']][2]
        if commission_value >= market_value:
            row['Direction'] = row['Direction']  # 可以在这里添加你的逻辑
        else:
            row['Direction'] = row['Direction'] * int(market_value / commission_value)
        return row
    
    def omit_drawback(row):
        commission_rate = commission_info[row['Asset']][2]
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