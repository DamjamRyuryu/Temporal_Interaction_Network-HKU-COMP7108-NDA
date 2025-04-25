import pandas as pd
import numpy as np
import os
import time as t

# 设置批处理大小和最大批次数来控制读取量
CHUNK_SIZE = 500000
# CHUNK_COUNT = 5

# 设置最终保留的记录数
MAX_RECORDS_TO_KEEP = 1000000

# 初始化统计数据
all_nodes = set()
all_edges = set()
total_interactions = 0
sum_passengers = 0
passenger_count = 0

# 创建输出文件
output_file = 'tin_graph.txt'
temp_file = 'tin_temp.txt'

# 记录开始时间
start_time = t.time()

print("Start batch processing...")

# 创建一个临时文件，用于存储所有交互记录
with open(temp_file, 'w') as f_temp:
    # 使用分块读取处理整个文件
    for chunk_number, chunk in enumerate(pd.read_csv('yellow_taxi.csv',
                                                  chunksize=CHUNK_SIZE,
                                                  low_memory=False)):
        chunk_start = t.time()
        print(f"Processing batch <{chunk_number + 1}> (Chunk size: {CHUNK_SIZE})...", end="")
        
        # 数据预处理：只保留需要的字段
        required_fields = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'passenger_count']
        
        # 确保所有需要的列都存在
        if not all(field in chunk.columns for field in required_fields):
            print(f"Warning：some records in this chunk miss required fields, skip.")
            continue
            
        chunk = chunk[required_fields].copy()
        
        # 清理缺失值
        chunk = chunk.dropna(subset=required_fields)
        
        # 转换日期时间格式
        chunk['tpep_pickup_datetime'] = pd.to_datetime(chunk['tpep_pickup_datetime'], errors='coerce')
        chunk = chunk.dropna(subset=['tpep_pickup_datetime'])  # 丢弃无效日期
        
        # 将时间转换为时间戳（整数）以简化输出
        chunk['pickup_timestamp'] = chunk['tpep_pickup_datetime'].astype(np.int64) // 10 ** 9
        
        # 更新统计信息
        current_nodes = set(chunk['PULocationID']).union(set(chunk['DOLocationID']))
        all_nodes.update(current_nodes)
        
        current_edges = set(zip(chunk['PULocationID'], chunk['DOLocationID']))
        all_edges.update(current_edges)
        
        batch_interactions = len(chunk)
        total_interactions += batch_interactions
        
        # 计算乘客总数
        valid_passengers = chunk['passenger_count'].dropna()
        sum_passengers += valid_passengers.sum()
        passenger_count += len(valid_passengers)
        
        # 写入TIN数据到临时文件，先不排序
        for _, row in chunk.iterrows():
            try:
                source = int(row['PULocationID'])
                dest = int(row['DOLocationID'])
                time = int(row['pickup_timestamp'])
                flow = int(row['passenger_count'])
                f_temp.write(f"{source}\t{dest}\t{time}\t{flow}\n")
            except (ValueError, TypeError) as e:
                print(f"Something wrong, skip 1 line: {e}")
                # 跳过无效数据
                continue
        
        chunk_end = t.time()
        print(f"completed in {chunk_end - chunk_start:.2f} seconds")

# 计算最终统计信息
num_nodes = len(all_nodes)
num_edges = len(all_edges)
avg_passengers = sum_passengers / passenger_count if passenger_count > 0 else 0

# 对整个数据集进行排序，并只保留前MAX_RECORDS_TO_KEEP条记录
print(f"Sorting all data and keeping only the first {MAX_RECORDS_TO_KEEP:,} records...")
sort_start = t.time()

# 读取所有行并排序
lines = []
with open(temp_file, 'r') as f:
    for line in f:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 3:  # 确保至少有源、目标和时间戳
            try:
                timestamp = int(parts[2])
                lines.append((timestamp, line))
            except (ValueError, IndexError):
                continue

# 对所有行按时间戳排序
print(f"Sorting {len(lines):,} lines...")
lines.sort()  # 按时间戳排序

# 只保留前MAX_RECORDS_TO_KEEP条记录
if len(lines) > MAX_RECORDS_TO_KEEP:
    print(f"Truncating to {MAX_RECORDS_TO_KEEP:,} lines...")
    lines = lines[:MAX_RECORDS_TO_KEEP]

# 更新总交互数为保留的记录数
total_interactions = len(lines)

sort_end = t.time()
print(f"Sorting and truncating completed in {sort_end - sort_start:.2f} seconds")

# 将最终信息写入正式输出文件
print("Generating TIN graph file...")
with open(output_file, 'w') as f_out:
    # 首先写入节点数和交互数
    f_out.write(f"{num_nodes}\n")
    f_out.write(f"{total_interactions}\n")
    
    # 然后写入排序后的TIN数据
    for _, line in lines:
        f_out.write(line)

# 删除临时文件
if os.path.exists(temp_file):
    os.remove(temp_file)
    print("Temp file removed.")

# 记录结束时间和总处理时间
end_time = t.time()
total_time = end_time - start_time

# 打印统计信息
print("\nDone！")
print(f"Time elapsed: {total_time:.2f} seconds")
print(f"Nodes: {num_nodes}")
print(f"Edges: {num_edges}")
print(f"Interactions: {total_interactions} (limited to first {MAX_RECORDS_TO_KEEP:,})")
print(f"Avg pax: {avg_passengers:.2f}")
print(f"TIN data saved to: {output_file}")
