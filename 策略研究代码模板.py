from multiprocessing import Pool, cpu_count
from 传统动量指标集 import *
from 传统波动率指标集 import *
from 国泰君安指标计算集 import *
import 计算函数包 as cf
from 世宽指标计算集 import *
from tqdm.contrib.concurrent import process_map
from 回测函数包 import *
from functools import partial
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial
from datetime import datetime

import os

os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'

if __name__ == "__main__":
    feature_compute_funcs = [
        high_low_5days, high_low_17days,

        volume_std_2days, volume_std_11days, volume_std_21days,

        returns_std_3days, returns_std_27days, returns_last_5days, returns_last_18day, returns_last_2days,

        weighted_avg_return_2, weighted_avg_return_42,

        returns_daliy_max_2days, returns_daliy_max_7days, returns_daliy_max_last_20days,

        sma_weight_con_23_37, sma_weight_con_3_62,

        flipping_weight_29_69, flipping_weight_4_67,

        gj_001, gj_002, gj_003, gj_004, gj_005, gj_006, gj_007, gj_008, gj_009, gj_010,
        gj_011, gj_012, gj_013, gj_014, gj_015, gj_016, gj_017, gj_018, gj_019, gj_020,
        gj_021, gj_022, gj_023, gj_024, gj_025, gj_026, gj_027, gj_028, gj_029, gj_031,
        gj_032, gj_033, gj_034, gj_035, gj_036, gj_037, gj_038, gj_039, gj_040, gj_041,
        gj_042, gj_043, gj_044, gj_045, gj_046, gj_047, gj_048, gj_049, gj_050, gj_051,
        gj_052, gj_053, gj_054, gj_056, gj_057, gj_058, gj_059, gj_060, gj_061, gj_062,
        gj_063, gj_064, gj_065, gj_066,

        wq_001, wq_002, wq_003, wq_004, wq_005, wq_006, wq_007, wq_008, wq_009, wq_010,
        wq_011, wq_012, wq_013, wq_014, wq_015, wq_016, wq_017, wq_018, wq_019, wq_020,
        wq_021, wq_022, wq_023, wq_024, wq_025, wq_026, wq_027, wq_028, wq_029, wq_030,
        wq_031, wq_032, wq_033, wq_034, wq_035, wq_036, wq_037, wq_038, wq_039, wq_040,
        wq_041, wq_042, wq_043, wq_044, wq_045, wq_046, wq_047, wq_049, wq_050, wq_051,
        wq_052, wq_053, wq_054, wq_055, wq_057, wq_065, wq_064, wq_060, wq_061, wq_066,
        wq_068, wq_070, wq_071, wq_072, wq_073, wq_074, wq_075, wq_077, wq_078, wq_081,
        wq_083, wq_085
    ]
    # 读取因子名
    feature_names = [func.__name__ for func in feature_compute_funcs]

    # 读取数据
    # df_path = r"C:\\Users\\Administrator\\Desktop\\人工智能多指标选期策略2.0\\人工智能多指标选期策略\\标记了换月日的已经连续化的日级数据.csv"
    # df = pd.read_csv(df_path)
    # #
    # # # 计算指标
    # df_unormal = cf.feature_compute(df, feature_compute_funcs)
    # # # 标准化
    # df = cf.truncated_normalize(df_unormal, feature_names)
    #
    # # 在第一次计算之后，建议将计算好指标的数据集存储起来，后面直接读取计算会节省很多时间
    # df.to_csv(r"C:\\Users\\Administrator\\Desktop\\人工智能多指标选期策略2.0\\人工智能多指标选期策略\\标记了换月日的已经连续化的日级数据指标已经计算(去未来）.csv")
    df = pd.read_csv("Dataset/指标已计算/dataset.csv")

    df['date'] = pd.to_datetime(df['date'])
    # print("特征计算已完成")

    # 删去换月前后两日以及周五，注意保留data这个原始数据集
    df = df.loc[df['huanyue'] != 1]
    #df = df.loc[df['huanyue'] != 2]
    df = df.loc[df['date'].dt.dayofweek != 4]
    print("已删除所有周五和所有换月日")

    # 根据斯皮尔曼系数以及缺失值多少进行指标筛选
    # ic = cf.information_coefficient(df, feature_list=feature_names)
    # features = cf.factor_filter_double(df, ic, number_bar=0.1, top_n=15, ic_bar=0.07125)
    # # print(features)
    # print("指标筛选已完成")

    # 读取期货id
    futures_name = df['id'].unique()
    diction_model = []
    for name in futures_name:
        # 读取指标名称
        '''changed'''
        selected_features = feature_names
        print(name)

        columns_to_extract = ['date'] + ['id'] + selected_features + ['returns']
        df_subset = df[df['id'] == name][columns_to_extract]

        # 删除缺失值
        df_subset = df_subset.dropna(thresh=int(len(df_subset)*0.9), axis=1)
        df_subset = df_subset.dropna()
        df_subset.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_subset.fillna(method='ffill', inplace=True)

        # 新建存储模型预测结果的列
        df_subset['prediction'] = 0
        df_subset['prediction'] = df_subset['prediction'].astype(float)

        # 将数据前
        threshold = int(0.99 * len(df_subset))
        if threshold <= 287:
            # 如果某品种过往数据太少，这个品种将不参与运算
            print("当前品种交易数据太少")
            continue
        else:
            df_subset['label'] = np.where(df_subset.index.isin(df_subset.index[:287]), -1, 1)

        # results.append(cf.linear_regression_rolling(df_subset, length=287))

        diction_model.append(df_subset)
    print("所有准备工作已完成，即将开始滚动训练")

    #指定所用样本点数
    partial_function = partial(cf.linear_regression_rolling, length=287)

    #使用多进程进行回测
    results = process_map(partial_function, diction_model, max_workers=cpu_count() - 1)
    prediction = pd.concat(results, ignore_index=True)
    print(prediction)
    print('滚动训练已结束,即将开始回测')

    # 读取未进行换月连续化的数据，提高回测的准确性
    data_path = r"Dataset/未连续化/标记了换月日的未连续化的日级数据.csv"
    data = pd.read_csv(data_path)
    data['date'] = pd.to_datetime(data['date'])
    result = data.merge(prediction[['id', 'date', 'label', 'prediction']], on=['id', 'date'], how='left')

    # 使用信号生成函数生成交易记录，此处设置阈值为0.005
    signals = cf.signals_day(result, 0.002)
    print(signals)
    # 使用等市值函数将开仓手数进行等市值处理
    signals = equal_weight(signals, 80000)
    timenow = datetime.now()
    timestamp_str = timenow.strftime("%Y-%m-%d_%H-%M-%S") 
    save_path = f"records/{timestamp_str}_开仓信号.csv"  
    signals.to_csv(save_path)
    # 输出回测结果
    backtest(800000, signals)
