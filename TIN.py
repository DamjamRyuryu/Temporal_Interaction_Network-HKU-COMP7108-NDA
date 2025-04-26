import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
import os
import time as t
import random
import numpy as np
import heapq
import copy

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
        # print(f"Nodes: {len(self.nodes)}, Edges: {len(self.edges)}, Interactions: {len(self.interactions)}")
    
    def create_from_taxi_data(self, taxi_data_path="yellow_taxi.csv"):
        """从出租车数据创建TIN
        
        参数:
            taxi_data_path: 出租车数据文件路径
        """
        # 如果预处理后的文件已存在，则直接加载
        if os.path.exists('graph.txt'):
            print("Existing file found.")
            self.load_from_file('graph.txt')
            return
        
        import subprocess
        print("Preprocessing data...")
        subprocess.run(['python', 'preprocess_taxi_data.py'])
        
        # 加载预处理后的数据
        self.load_from_file('tin_graph.txt')
        
    
    def get_statistics(self):
        """获取网络统计信息"""
        # 计算每次行程的平均乘客数
        total_passengers = sum(flow for _, _, _, flow in self.interactions)
        avg_passengers = total_passengers / len(self.interactions) if self.interactions else 0
        
        # 构建统计信息字典
        stats = {
            "Nodes": len(self.nodes),
            "Edges": len(self.edges),
            "Interactions": len(self.interactions),
            "Avg_pax": avg_passengers
        }
        return stats

    
    def visualize(self, with_interactions=True, max_edges=1000, output_file='tin_network.png'):
        """可视化TIN网络
        
        参数:
            with_interactions: 是否显示边的交互数量
            max_edges: 可视化的最大边数
            output_file: 输出图像文件名
        """
        print("\nPreparing to visualize network...")
        start_time = t.time()
        
        # 直接使用self.graph
        G = self.graph.copy()
        
        # 如果边数过多，选择最重要的边
        if G.number_of_edges() > max_edges:
            print(f"Edges ({G.number_of_edges()}) exceed max limit ({max_edges}), select edges with more interactions.")
            
            # 计算每条边的交互次数作为重要性指标
            edge_importance = {}
            for u, v, data in G.edges(data=True):
                interactions = data.get('interactions', [])
                edge_importance[(u, v)] = len(interactions)
            
            # 选择交互次数最多的边
            important_edges = sorted(edge_importance.items(), key=lambda x: x[1], reverse=True)[:max_edges]
            
            # 创建一个新图，只包含重要的边
            important_G = nx.DiGraph()
            for (u, v), _ in important_edges:
                important_G.add_edge(u, v, **G[u][v])
            
            G = important_G
        
        # 获取实际使用的节点
        nodes_to_draw = set(G.nodes())
        
        print(f"Visualizing {len(nodes_to_draw)} nodes and {G.number_of_edges()} edges")
        
        # 绘制图
        plt.figure(figsize=(16, 12))
        
        # 使用spring布局
        print("Computing node layout...")
        pos = nx.spring_layout(G, k=0.1, iterations=50)
        
        # 绘制节点
        print("Drawing nodes...")
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue', alpha=1.0)
        
        # 绘制边
        print("Drawing edges...")
        if with_interactions:
            # 为每条边设置宽度，基于交互次数
            edge_widths = []
            for u, v in G.edges():
                interactions = G[u][v].get('interactions', [])
                edge_widths.append(len(interactions))
            
            # 归一化边宽度
            if edge_widths:
                max_interactions = max(edge_widths) if edge_widths else 1
                min_width, max_width = 0.5, 5.0
                normalized_widths = [min_width + (count / max_interactions) * (max_width - min_width) if max_interactions > 0 else min_width for count in edge_widths]
                
                # 检测相互连接的边并使用不同的弧度
                mutual_edges = {}
                for u, v in G.edges():
                    if G.has_edge(v, u):  # 如果存在相反方向的边
                        if (v, u) not in mutual_edges:  # 避免重复处理
                            mutual_edges[(u, v)] = 0.3  # 正向边使用正弧度
                            mutual_edges[(v, u)] = 0.3  # 反向边使用正弧度
                
                # 绘制边，为相互连接的边使用特定的弧度
                edge_list = list(G.edges())
                for i, (u, v) in enumerate(edge_list):
                    rad = mutual_edges.get((u, v), 0.1)  # 默认弧度为0.1
                    nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=normalized_widths[i], 
                                         alpha=0.6, edge_color='gray', arrowsize=15, 
                                         connectionstyle=f'arc3,rad={rad}')
            else:
                nx.draw_networkx_edges(G, pos, alpha=0.6, edge_color='gray', 
                                     arrowsize=15, connectionstyle='arc3,rad=0.1')
        else:
            # 检测相互连接的边并使用不同的弧度
            mutual_edges = {}
            for u, v in G.edges():
                if G.has_edge(v, u):  # 如果存在相反方向的边
                    if (v, u) not in mutual_edges:  # 避免重复处理
                        mutual_edges[(u, v)] = 0.3  # 正向边使用正弧度
                        mutual_edges[(v, u)] = 0.3  # 反向边使用正弧度
            
            # 使用不同的弧度绘制边
            for u, v in G.edges():
                rad = mutual_edges.get((u, v), 0.1)  # 默认弧度为0.1
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=0.6, edge_color='gray', 
                                     arrowsize=15, connectionstyle=f'arc3,rad={rad}')
        
        # 绘制节点标签
        print("Drawing node labels...")
        # 如果节点太多，只显示部分节点标签
        if len(nodes_to_draw) > 50:
            print("Too many nodes, displaying only some labels")
            # 选择度最高的节点显示标签
            nodes_with_labels = sorted(G.degree, key=lambda x: x[1], reverse=True)[:50]
            labels = {node: str(node) for node, _ in nodes_with_labels}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='sans-serif', verticalalignment='baseline')
        else:
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', verticalalignment='baseline')
        
        # 如果边不太多，显示边上的interactions
        if G.number_of_edges() < 100:  # 减少显示标签的边数，因为交互列表可能很长
            print("Drawing interactions on edges...")
            edge_labels = {}
            for u, v in G.edges():
                interactions = G[u][v].get('interactions', [])
                # 限制显示的交互数量，最多显示3个
                display_interactions = interactions[:3]
                if interactions:
                    label = ""
                    for i, (time, flow) in enumerate(display_interactions):
                        label += f"({time},{flow})"
                        if i < len(display_interactions) - 1:
                            label += ", "
                    if len(interactions) > 3:
                        label += f"... +{len(interactions)-3} more"
                    edge_labels[(u, v)] = label
            
            # 绘制边标签，使用自定义位置
            nx.draw_networkx_edge_labels(
                G, pos, 
                edge_labels=edge_labels,
                label_pos=0.2,
                font_size=8, 
                font_weight='bold')
        
        plt.title('Temporal Interaction Network')
        plt.axis('off')
        plt.tight_layout()
        
        # 保存图像
        print(f"Saving image to {output_file}...")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        end_time = t.time()
        print(f"Visualization completed. Time elapsed: {end_time - start_time:.2f} seconds")
    
    def track_provenance_lrb(self, target_node=None):
        """
        使用最旧优先(Least Recently Born, LRB)策略追踪数据来源
        
        参数:
            target_node: 目标节点，如果为None则追踪所有节点
            
        返回:
            节点的数据量缓冲区字典
        """
        print("使用LRB策略追踪数据来源...")
        start_time = t.time()
        
        # 按时间排序所有交互
        sorted_interactions = sorted(self.interactions, key=lambda x: x[2])
        
        # 初始化每个节点的缓冲区 (最小堆，按时间排序)
        buffers = {node: [] for node in self.nodes}
        
        # 处理所有交互
        for i, (source, dest, time, quantity) in enumerate(sorted_interactions):
            if i % 100000 == 0 and i > 0:
                print(f"已处理 {i} 个交互...")
            
            # 剩余需要转移的数据量
            remaining_quantity = quantity
            
            # 当有剩余数据量且源节点缓冲区非空时，继续处理
            while remaining_quantity > 0 and buffers[source]:
                # 获取最早生成的三元组（不弹出）
                oldest_triple = buffers[source][0]
                origin, birth_time, avail_quantity = oldest_triple
                
                if avail_quantity > remaining_quantity:
                    # 分割三元组
                    # 更新源节点中的三元组
                    heapq.heapreplace(buffers[source], (origin, birth_time, avail_quantity - remaining_quantity))
                    # 添加到目标节点
                    heapq.heappush(buffers[dest], (origin, birth_time, remaining_quantity))
                    remaining_quantity = 0
                else:
                    # 移除整个三元组
                    heapq.heappop(buffers[source])
                    # 添加到目标节点
                    heapq.heappush(buffers[dest], (origin, birth_time, avail_quantity))
                    remaining_quantity -= avail_quantity
            
            # 如果还有剩余数据量，生成新的三元组
            if remaining_quantity > 0:
                heapq.heappush(buffers[dest], (source, time, remaining_quantity))
        
        end_time = t.time()
        print(f"数据来源追踪完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 如果指定了目标节点，只返回该节点的缓冲区
        if target_node is not None:
            if target_node in buffers:
                result = buffers[target_node]
                print(f"节点 {target_node} 的缓冲区包含 {len(result)} 个三元组")
                return result
            else:
                print(f"节点 {target_node} 不存在")
                return []
        
        return buffers
    
    def track_provenance_mrb(self, target_node=None):
        """
        使用最新优先(Most Recently Born, MRB)策略追踪数据来源
        
        参数:
            target_node: 目标节点，如果为None则追踪所有节点
            
        返回:
            节点的数据量缓冲区字典
        """
        print("使用MRB策略追踪数据来源...")
        start_time = t.time()
        
        # 按时间排序所有交互
        sorted_interactions = sorted(self.interactions, key=lambda x: x[2])
        
        # 初始化每个节点的缓冲区
        # 对于MRB，我们使用最大堆，但heapq是最小堆，所以存储负的时间戳
        buffers = {node: [] for node in self.nodes}
        
        # 处理所有交互
        for i, (source, dest, time, quantity) in enumerate(sorted_interactions):
            if i % 100000 == 0 and i > 0:
                print(f"已处理 {i} 个交互...")
            
            # 剩余需要转移的数据量
            remaining_quantity = quantity
            
            # 当有剩余数据量且源节点缓冲区非空时，继续处理
            while remaining_quantity > 0 and buffers[source]:
                # 获取最新生成的三元组（不弹出）
                # 注意：存储的是(-birth_time, origin, avail_quantity)
                newest_triple = buffers[source][0]
                neg_birth_time, origin, avail_quantity = newest_triple
                birth_time = -neg_birth_time  # 还原实际时间
                
                if avail_quantity > remaining_quantity:
                    # 分割三元组
                    # 更新源节点中的三元组
                    heapq.heapreplace(buffers[source], (neg_birth_time, origin, avail_quantity - remaining_quantity))
                    # 添加到目标节点
                    heapq.heappush(buffers[dest], (neg_birth_time, origin, remaining_quantity))
                    remaining_quantity = 0
                else:
                    # 移除整个三元组
                    heapq.heappop(buffers[source])
                    # 添加到目标节点
                    heapq.heappush(buffers[dest], (neg_birth_time, origin, avail_quantity))
                    remaining_quantity -= avail_quantity
            
            # 如果还有剩余数据量，生成新的三元组
            if remaining_quantity > 0:
                heapq.heappush(buffers[dest], (-time, source, remaining_quantity))  # 注意负时间
        
        end_time = t.time()
        print(f"数据来源追踪完成，耗时: {end_time - start_time:.2f} 秒")
        
        # 如果指定了目标节点，只返回该节点的缓冲区
        if target_node is not None:
            if target_node in buffers:
                # 转换结果，使时间为正值
                result = [(origin, -neg_time, quantity) for neg_time, origin, quantity in buffers[target_node]]
                print(f"节点 {target_node} 的缓冲区包含 {len(result)} 个三元组")
                return result
            else:
                print(f"节点 {target_node} 不存在")
                return []
        
        # 转换所有结果，使时间为正值
        converted_buffers = {}
        for node, buffer in buffers.items():
            converted_buffers[node] = [(origin, -neg_time, quantity) for neg_time, origin, quantity in buffer]
        
        return converted_buffers
    
    def analyze_provenance(self, buffer, k=10):
        """
        分析追踪结果，返回数据来源统计
        
        参数:
            buffer: 缓冲区（三元组列表）
            k: 返回前k个来源
            
        返回:
            节点及其贡献的字典，按贡献降序排序
        """
        if not buffer:
            return {}
        
        # 统计每个来源的贡献量
        origin_stats = defaultdict(float)
        for origin, _, quantity in buffer:
            origin_stats[origin] += quantity
            
        # 按贡献量降序排序
        sorted_stats = sorted(origin_stats.items(), key=lambda x: x[1], reverse=True)
        
        # 返回前k个来源
        return dict(sorted_stats[:k])


if __name__ == "__main__":
    # 创建TIN实例
    tin = TemporalInteractionNetwork()
    tin.create_from_taxi_data('yellow_taxi.csv')
    
    # 输出统计信息
    stats = tin.get_statistics()
    print("\nNetwork statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    tin.visualize()
    # # 随机选择一个目标节点
    # target_node = random.choice(list(tin.nodes))
    # print(f"\n随机选择的目标节点: {target_node}")
    #
    # # 使用LRB策略追踪数据来源
    # print("\n使用LRB策略（最旧优先）追踪数据来源...")
    # lrb_buffer = tin.track_provenance_lrb(target_node)
    # lrb_stats = tin.analyze_provenance(lrb_buffer)
    #
    # # 显示LRB结果
    # print("\nLRB策略（最旧优先）前10个来源:")
    # for origin, quantity in sorted(lrb_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
    #     print(f"节点 {origin}: {quantity:.2f} 单位")
    #
    # # 可视化LRB结果
    # tin.visualize_provenance(lrb_stats, title=f"LRB策略（最旧优先）- 节点{target_node}的数据来源",
    #                         output_file=f"provenance_lrb_{target_node}.png")
    #
    # # 使用MRB策略追踪数据来源
    # print("\n使用MRB策略（最新优先）追踪数据来源...")
    # mrb_buffer = tin.track_provenance_mrb(target_node)
    # mrb_stats = tin.analyze_provenance(mrb_buffer)
    #
    # # 显示MRB结果
    # print("\nMRB策略（最新优先）前10个来源:")
    # for origin, quantity in sorted(mrb_stats.items(), key=lambda x: x[1], reverse=True)[:10]:
    #     print(f"节点 {origin}: {quantity:.2f} 单位")
    #
    # # 可视化MRB结果
    # tin.visualize_provenance(mrb_stats, title=f"MRB策略（最新优先）- 节点{target_node}的数据来源",
    #                         output_file=f"provenance_mrb_{target_node}.png")
    #
    # print("\n完成!")