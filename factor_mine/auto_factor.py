
# 自动挖掘因子

# 1. 环境配置
print('1. 环境配置')

import os
import pandas as pd
import numpy as np
from tqdm import tqdm

# 2. 数据读取
print('2. 数据读取')

factors = {}  # 因子

# 共同因子读取(factor1~factor26)
data_dir = '/Quant/factor_readable/com_factor'  # 数据目录(共同因子)

for csv_name in tqdm(sorted(os.listdir(data_dir)), desc="共同因子"):
    if csv_name.endswith('.csv'):
        data_path = os.path.join(data_dir, csv_name)  # 数据路径
        data = pd.read_csv(data_path)  # 数据
        data = data.rename(columns={data.columns[0]: 'index'})  # 修改第1列列名为'index'
        data = data.set_index(data.columns[0])  # 设置第1列为索引
        factor_name = csv_name[:-4]  # 因子名称，'factor1'
        factor_index = int(factor_name[6:])  # 因子索引，1
        factors[factor_index] = data  # 因子

# 基础因子读取(factor27~factor34)
data_dir = '/Quant/factor_readable/base_factor'  # 数据目录(基础因子)

for csv_name in tqdm(sorted(os.listdir(data_dir)), desc="基础因子"):
    if csv_name.endswith('.csv'):
        data_path = os.path.join(data_dir, csv_name)  # 数据路径
        data = pd.read_csv(data_path)  # 数据
        data = data.rename(columns={data.columns[0]: 'index'})  # 修改第1列列名为'index'
        data = data.set_index(data.columns[0])  # 设置第1列为索引
        factor_name = csv_name[:-4]  # 因子名称，'factor1'
        factor_index = int(factor_name[6:])  # 因子索引，1
        factors[factor_index] = data  # 因子

# 额外因子读取(factor35~factor41)
data_dir = '/Quant/factor_readable/extra_factor'  # 数据目录(额外因子)

for csv_name in tqdm(sorted(os.listdir(data_dir)), desc="额外因子"):
    if csv_name.endswith('.csv'):
        data_path = os.path.join(data_dir, csv_name)  # 数据路径
        data = pd.read_csv(data_path)  # 数据
        data = data.rename(columns={data.columns[0]: 'index'})  # 修改第1列列名为'index'
        data = data.set_index(data.columns[0])  # 设置第1列为索引
        factor_name = csv_name[:-4]  # 因子名称，'factor1'
        factor_index = int(factor_name[6:])  # 因子索引，1
        factors[factor_index] = data  # 因子

# 示例因子读取(factor42~factor47)
data_dir = '/Quant/factor_readable/eg_factor'  # 数据目录(示例因子)

for csv_name in tqdm(sorted(os.listdir(data_dir)), desc="示例因子"):
    if csv_name.endswith('.csv'):
        data_path = os.path.join(data_dir, csv_name)  # 数据路径
        data = pd.read_csv(data_path)  # 数据
        data = data.rename(columns={data.columns[0]: 'index'})  # 修改第1列列名为'index'
        data = data.set_index(data.columns[0])  # 设置第1列为索引
        factor_name = csv_name[:-4]  # 因子名称，'factor1'
        factor_index = int(factor_name[6:])  # 因子索引，1
        factors[factor_index] = data  # 因子
        
factor_num = len(factors)  # 因子数目
future_names = factors[1].columns  # 期货名称  # TODO
future_num = len(future_names)  # 期货数目

print('因子数目:', factor_num)
print('期货名称:', future_names.values)
print('期货数目:', future_num)

print('shape:', factors[1].shape)  # TODO

# 因子预处理
for factor_key, factor_value in factors.items():  # factor_key=1
    data = factor_value
    data = data.fillna(0.00)  # 填充缺失值
    data.index = pd.to_datetime(data.index, format='%Y%m%d')  # 转换索引为datetime格式
    data = data.reset_index().rename(columns={'index': 'date'})  # 修改索引列名为'date'
    factors[factor_key] = data  # 因子

FACTORS = {}  # ! 因子，键为期货，值为因子

for future_name in future_names:  # 期货名称，'A'
    data = pd.DataFrame()
    data['date'] = factors[1]['date']  # TODO: 日期，2013-01-04
    row_num = len(data)  # 行数，2430
    data['code'] = pd.Series([future_name] row_num)  # 期货名称，'A'
    
    for factor_key, factor_value in factors.items():  # factor_key=1
        data[f'factor{factor_key}'] = factor_value[future_name]
    
    FACTORS[future_name] = data  # 因子
    
print('shape:', FACTORS['A'].shape)

# 3. 准备数据
print('3. 准备数据')

# 滚动(roll)变换
from tsfresh.utilities.dataframe_functions import roll_time_series

data_roll = {}  # 滚动数据

for future_name in future_names:  # 期货名称，'A'
    print('future:', future_name)  # 期货名称
    data_roll[future_name] = roll_time_series(FACTORS[future_name], column_id='code', column_sort='date', max_timeshift=20, min_timeshift=5).drop(columns=['code'])  # 滚动数据

print('shape:', data_roll['A'].shape)

# 换种方式展示
gg = {}  # 期货数据

for future_name in future_names:  # 期货名称，'A'
    gg[future_name] = data_roll[future_name].groupby('id').agg({'date':['count', min, max]})  # 期货数据
 
print('shape:', gg['A'].shape)

# 衍生出众多因子
from tsfresh import extract_features

data_feat = {}  # 数据特征

for future_name in future_names:  # 期货名称，'A'
    print('future:', future_name)
    data_feat[future_name] = extract_features(data_roll[future_name], column_id='id', column_sort='date')  # 数据特征
    # 对单独标的而言，将日期作为index
    data_feat[future_name].index = [v[1] for v in data_feat[future_name].index]

auto_factor_num = data_feat['A'].shape[1]  # 自动因子数目
print('shape:', data_feat['A'].shape)
print('自动因子数目:', auto_factor_num)

# 4. 准备数据
print('4. 准备数据')

# 填充日期
data_dir = '/Quant/factor_readable/com_factor'  # 数据目录(共同因子)
data_path = os.path.join(data_dir, 'factor1.csv')  # 数据路径
data_df = pd.read_csv(data_path)
data_df = data_df.rename(columns={data_df.columns[0]: 'index'})  # 修改第1列列名为'index'
data_df = data_df.set_index(data_df.columns[0])  # 设置第1列为索引
data_df.index = pd.to_datetime(data_df.index, format='%Y%m%d')  # 转换索引为datetime格式
data_df = data_df.reset_index().rename(columns={'index': 'date'})  # 修改index'列名为'date'
data_sr = data_df['date']  # 'date'列
data = pd.DataFrame(np.nan, index=data_sr[:5], columns=data_feat['A'].columns)  # factor的前5行

print('shape:', data.shape)

# 数据拼接
for future_name in future_names:  # 期货名称，'A'
    data_feat[future_name] = pd.concat([data, data_feat[future_name]])  # 拼接数据特征

print('shape:', data_feat['A'].shape)

# 5. 数据导出
print('5. 数据导出')

for future_name in future_names:  # 期货名称，'A'
    data_feat[future_name] = data_feat[future_name].reset_index().rename(columns={'index': 'date'})  # # 修改'index'列名为'date'

data_dir = '/Quant/factor_readable/com_factor'  # 数据目录(共同因子)
data_path = os.path.join(data_dir, 'factor1.csv')  # 数据路径
data_df = pd.read_csv(data_path)
date_sr = data_df['index']  # 日期列
        
data_dir = '/Quant/factor_readable/auto_factor'  # 数据目录(自动因子)
begin_index = 48

for factor_index in tqdm(range(begin_index, begin_index + auto_factor_num), desc="因子文件导出"):
    data = pd.DataFrame()
    data['index'] = date_sr  # 日期，20130104

    for future_name in future_names:  # 期货名称，'A'
        factor_name = data_feat[future_name].columns[factor_index-begin_index+1]  # 'factor?'
        data[future_name] = data_feat[future_name][factor_name]  # 因子
    
    data = data.rename(columns={data.columns[0]: 'index'})  # 修改第1列列名为'index'
    data = data.set_index(data.columns[0])  # 设置第1列为索引
    factors[factor_index] = data  # 因子
    
    factors[factor_index].to_csv(f'{data_dir}/factor{factor_index}.csv')  # 文件导出

print('shape:', factors[begin_index].shape)
