import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time as t
import random

class TemporalInteractionNetwork:
    def __init__(self):
        """初始化时间交互网络"""
        self.nodes = set()  # 节点集合
        self.edges = set()  # 边集合（源节点和目标节点的唯一对）
        self.interactions = []  # 交互列表（源，目标，时间，流量）
        self.graph = nx.DiGraph()  # 有向图表示
        
    def load_from_file(self, file_path='tin_graph.txt'):
        """从文件加载TIN数据
        
        参数:
            file_path: TIN数据文件路径
        """
        print(f"Loading TIN data from {file_path}...")
        start_time = t.time()
        
        # 首先读取前两行获取节点数和交互数
        with open(file_path, 'r') as f:
            num_nodes = int(f.readline().strip())
            num_interactions = int(f.readline().strip())
            
            print(f"This TIN contains {num_nodes} nodes and {num_interactions} interactions.")
            
            # 逐行读取交互数据
            counter = 0
            for line in f:
                if not line.strip():
                    continue
                    
                parts = line.strip().split('\t')
                if len(parts) < 4:
                    print(f"Warning: skipping an invalid line: {line}")
                    continue
                    
                source, dest, time, flow = map(int, parts[:4])
                self.nodes.add(source)
                self.nodes.add(dest)
                self.edges.add((source, dest))
                self.interactions.append((source, dest, time, flow))
                
                # 更新图的边，使用边的属性存储交互列表
                if not self.graph.has_edge(source, dest):
                    self.graph.add_edge(source, dest, interactions=[])
                
                self.graph[source][dest]['interactions'].append((time, flow))
                
                counter += 1
                if counter % 100000 == 0:
                    print(f"{counter} interactions loaded.")
        
        end_time = t.time()
        print(f"Loading complete. Time elapsed : {end_time - start_time:.2f} seconds")
        print(f"Nodes: {len(self.nodes)}, Edges: {len(self.edges)}, Interactions: {len(self.interactions)}")
    
    def create_from_taxi_data(self, taxi_data_path, process_all=True):
        """从出租车数据创建TIN
        
        参数:
            taxi_data_path: 出租车数据文件路径
            process_all: 是否处理全部数据，True表示处理全部，False表示处理前100,000条
        """
        # 如果预处理后的文件已存在，则直接加载
        if os.path.exists('tin_output.txt'):
            print("发现已有TIN输出文件，直接加载...")
            self.load_from_file('tin_output.txt')
            return
        
    
    def get_statistics(self):
        """获取网络统计信息"""
        # 计算每次行程的平均乘客数
        total_passengers = sum(flow for _, _, _, flow in self.interactions)
        avg_passengers = total_passengers / len(self.interactions) if self.interactions else 0
        
        # 构建统计信息字典
        stats = {
            "节点数量": len(self.nodes),
            "边数量": len(self.edges),
            "交互数量": len(self.interactions),
            "每次行程的平均乘客数": avg_passengers
        }
        return stats
    
    def find_repeating_patterns(self, time_window=3600, sample_size=None):
        """查找重复模式
        
        参数:
            time_window: 时间窗口大小（秒）
            sample_size: 用于分析的样本大小，None表示使用全部数据
        返回:
            重复模式字典
        """
        print(f"查找重复模式 (时间窗口: {time_window}秒)...")
        start_time = t.time()
        
        # 如果数据很大，可以选择采样
        interactions_to_analyze = self.interactions
        if sample_size and sample_size < len(self.interactions):
            print(f"数据量较大，使用 {sample_size} 个随机样本进行分析")
            interactions_to_analyze = random.sample(self.interactions, sample_size)
        
        patterns = defaultdict(list)
        
        # 对交互按时间排序
        sorted_interactions = sorted(interactions_to_analyze, key=lambda x: x[2])
        
        # 查找在时间窗口内重复的边
        for source, dest, time, flow in sorted_interactions:
            edge = (source, dest)
            patterns[edge].append((time, flow))
        
        # 过滤出重复出现的模式
        repeating_patterns = {edge: times for edge, times in patterns.items() if len(times) > 1}
        
        # 进一步分析时间窗口内的重复
        time_window_patterns = {}
        pattern_count = 0
        
        for edge, times in repeating_patterns.items():
            windows = []
            current_window = [times[0]]
            
            for i in range(1, len(times)):
                if times[i][0] - current_window[-1][0] <= time_window:
                    current_window.append(times[i])
                else:
                    if len(current_window) > 1:
                        windows.append(current_window)
                    current_window = [times[i]]
            
            if len(current_window) > 1:
                windows.append(current_window)
            
            if windows:
                time_window_patterns[edge] = windows
                pattern_count += len(windows)
        
        end_time = time.time()
        print(f"模式识别完成。耗时: {end_time - start_time:.2f} 秒")
        print(f"在 {len(repeating_patterns)} 条边中找到 {pattern_count} 个重复模式")
        
        return time_window_patterns
    
    def track_quantity_provenance(self, target_node, time_limit=None, sample_size=None):
        """追踪在网络中转移的数量来源
        
        参数:
            target_node: 目标节点
            time_limit: 可选的时间限制
            sample_size: 分析的样本大小
        返回:
            源节点及其贡献的字典
        """
        print(f"追踪到节点 {target_node} 的数量来源...")
        start_time = t.time()
        
        # 如果数据很大，可以选择采样
        interactions_to_analyze = self.interactions
        if sample_size and sample_size < len(self.interactions):
            print(f"数据量较大，使用 {sample_size} 个随机样本进行分析")
            interactions_to_analyze = random.sample(self.interactions, sample_size)
        
        # 按时间排序交互
        sorted_interactions = sorted(interactions_to_analyze, key=lambda x: x[2])
        
        # 用于存储每个节点当前的数量
        node_quantities = defaultdict(int)
        
        # 用于追踪数量的来源
        provenance = defaultdict(lambda: defaultdict(int))
        
        # 处理所有交互
        for source, dest, time, flow in sorted_interactions:
            if time_limit and time > time_limit:
                break
                
            if flow > 0:
                # 更新目标节点接收的数量
                node_quantities[dest] += flow
                
                # 如果源节点有数量，则认为这是直接传输
                provenance[dest][source] += flow
                
                # 如果这是流向我们关注的目标节点的交互
                if dest == target_node:
                    # 已经在provenance中记录了source的贡献
                    pass
        
        end_time = time.time()
        print(f"来源追踪完成。耗时: {end_time - start_time:.2f} 秒")
        
        # 返回与目标节点相关的来源
        result = dict(provenance[target_node])
        print(f"找到 {len(result)} 个来源节点")
        
        return result
    
    def visualize(self, with_weights=True, max_edges=1000, output_file='tin_network.png'):
        """可视化TIN网络
        
        参数:
            with_weights: 是否显示边的权重
            max_edges: 可视化的最大边数
            output_file: 输出图像文件名
        """
        print("准备可视化网络...")
        start_time = t.time()
        
        # 创建一个新图，边的权重为交互次数
        G = nx.DiGraph()
        
        # 计算边权重
        edge_weights = defaultdict(int)
        for source, dest, _, flow in self.interactions:
            edge_weights[(source, dest)] += flow
        
        # 如果边数过多，选择权重最高的边
        if len(edge_weights) > max_edges:
            print(f"边数 ({len(edge_weights)}) 超过最大限制 ({max_edges})，选择权重最高的边")
            edge_weights = dict(sorted(edge_weights.items(), key=lambda x: x[1], reverse=True)[:max_edges])
        
        # 添加边和节点到图
        for (source, dest), weight in edge_weights.items():
            G.add_edge(source, dest, weight=weight)
        
        # 获取实际使用的节点
        nodes_to_draw = set()
        for source, dest in G.edges():
            nodes_to_draw.add(source)
            nodes_to_draw.add(dest)
        
        print(f"可视化 {len(nodes_to_draw)} 个节点和 {len(G.edges())} 条边")
        
        # 绘制图
        plt.figure(figsize=(16, 12))
        
        # 使用spring布局
        print("计算节点布局...")
        pos = nx.spring_layout(G, k=0.3, iterations=50)
        
        # 绘制节点
        print("绘制节点...")
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue', alpha=0.8)
        
        # 绘制边，边的宽度与权重成正比
        print("绘制边...")
        if with_weights:
            # 归一化边权重，使其在可视范围内
            max_weight = max(G[u][v]['weight'] for u, v in G.edges())
            min_width, max_width = 0.5, 5.0
            
            edge_width = [min_width + (G[u][v]['weight'] / max_weight) * (max_width - min_width) 
                          for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_width, alpha=0.6, edge_color='gray', 
                                 arrowsize=15, connectionstyle='arc3,rad=0.1')
        else:
            nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray', 
                                 arrowsize=15, connectionstyle='arc3,rad=0.1')
        
        # 绘制节点标签
        print("绘制节点标签...")
        # 如果节点太多，只显示部分节点标签
        if len(nodes_to_draw) > 50:
            print("节点太多，只显示部分节点标签")
            # 选择度最高的节点显示标签
            nodes_with_labels = sorted(G.degree, key=lambda x: x[1], reverse=True)[:50]
            labels = {node: str(node) for node, _ in nodes_with_labels}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='sans-serif')
        else:
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
        
        # 如果边不太多并且需要显示权重，显示边标签
        if len(G.edges()) < 50 and with_weights:
            print("绘制边标签...")
            edge_labels = {(u, v): f"{G[u][v]['weight']}" for u, v in G.edges()}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title('时间交互网络')
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图像
        print(f"保存图像到 {output_file}...")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        end_time = t.time()
        print(f"可视化完成。耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    # 创建TIN实例
    tin = TemporalInteractionNetwork()
    
    # 从出租车数据创建TIN
    # 参数process_all=True表示处理全部数据，False表示仅处理前100,000条
    print("您希望处理全部数据还是仅处理部分数据?")
    print("1. 处理全部数据 (可能需要较长时间)")
    print("2. 只处理前100,000条数据 (更快)")
    
    choice = input("请输入选项 (1 或 2): ").strip()
    process_all = (choice == '1')
    
    tin.create_from_taxi_data('yellow_taxi.csv', process_all=process_all)
    
    # 输出统计信息
    stats = tin.get_statistics()
    print("\n网络统计信息:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
    
    # 查找重复模式
    # 如果数据量很大，可以用采样来加快处理速度
    sample_size = 1000000 if len(tin.interactions) > 1000000 else None
    print("\n寻找重复模式...")
    patterns = tin.find_repeating_patterns(sample_size=sample_size)
    
    # 输出部分重复模式示例
    if patterns:
        print("\n重复模式示例（最多5个）:")
        for i, (edge, windows) in enumerate(list(patterns.items())[:5]):
            print(f"模式 {i+1}: {edge}出现在以下时间窗口中:")
            for j, window in enumerate(windows[:3]):  # 每个模式最多显示3个窗口
                print(f"  窗口 {j+1}: {len(window)}次交互，时间范围: {window[0][0]}-{window[-1][0]}")
    
    # 追踪来源示例
    if tin.nodes:
        target_node = random.choice(list(tin.nodes))
        print(f"\n追踪示例: 追踪到节点 {target_node} 的数量来源")
        
        # 对于大数据集，使用采样
        sample_size = 1000000 if len(tin.interactions) > 1000000 else None
        sources = tin.track_quantity_provenance(target_node, sample_size=sample_size)
        
        # 显示前10个主要来源
        top_sources = sorted(sources.items(), key=lambda x: x[1], reverse=True)[:10]
        print("\n前10个主要来源:")
        for source, amount in top_sources:
            print(f"节点 {source}: {amount} 单位")
    
    # 可视化网络
    print("\n正在可视化网络...")
    tin.visualize(max_edges=500)  # 限制最大显示的边数
    
    print("\n完成!")