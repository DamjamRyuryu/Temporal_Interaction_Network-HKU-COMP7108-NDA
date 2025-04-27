import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from preprocess_taxi_data import preprocess_taxi_data
import os
import time as t
import random
import heapq
import copy

PROVENANCE_POLICIES = ['lrb', 'mrb']


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
        print(f"Loading complete. Time elapsed : {end_time - start_time:.6f} seconds")
        # print(f"Nodes: {len(self.nodes)}, Edges: {len(self.edges)}, Interactions: {len(self.interactions)}")

    def create_from_taxi_data(self, graph_data='tin_graph.txt'):
        """从出租车数据创建TIN
        
        参数:
            taxi_data_path: 出租车数据文件路径
        """
        # 如果预处理后的文件已存在，则直接加载
        if os.path.exists(graph_data):
            print("Pre-generated file found.")
            self.load_from_file(graph_data)
            return

        choice = input("\nNo pre-generated file, process the taxi data" +
                       "\nDo you want to use limited (1 million) interactions? (otherwise use All interactions)\n[(Y)/n]=>")
        print("Preprocessing data...")
        preprocess_taxi_data(limited=False if choice.lower() == 'n' else True)

        # 加载预处理后的数据
        self.load_from_file('tin_graph.txt')

    def get_statistics(self):
        """获取网络统计信息"""
        # 计算每次行程的平均乘客数
        total_passengers = sum(flow for _, _, _, flow in self.interactions)
        avg_passengers = total_passengers / len(self.interactions) if self.interactions else 0
        max_ts = max(ts for _, _, ts, _ in self.interactions)
        min_ts = min(ts for _, _, ts, _ in self.interactions)

        # 构建统计信息字典
        stats = {
            "Nodes": len(self.nodes),
            "Edges": len(self.edges),
            "Interactions": len(self.interactions),
            "Avg_pax": avg_passengers,
            "min_timestamp": min_ts,
            "max_timestamp": max_ts
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
        print("Computing node layout...", end='')
        pos = nx.spring_layout(G, k=0.1, iterations=50)

        # 绘制节点
        print("Drawing nodes...", end='')
        nx.draw_networkx_nodes(G, pos, node_size=600, node_color='lightblue', alpha=1.0)

        # 绘制边
        print("Drawing edges...", end='')
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
                normalized_widths = [min_width + (count / max_interactions) * (
                        max_width - min_width) if max_interactions > 0 else min_width for count in edge_widths]

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
                    rad = mutual_edges.get((u, v), 0)  # 默认弧度为0
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
                rad = mutual_edges.get((u, v), 0)  # 默认弧度为0
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], alpha=0.6, edge_color='gray',
                                       arrowsize=15, connectionstyle=f'arc3,rad={rad}')

        # 绘制节点标签
        print("Drawing node labels...", end='')
        # 如果节点太多，只显示部分节点标签
        if len(nodes_to_draw) > 50:
            print("Too many nodes, displaying only some labels")
            # 选择度最高的节点显示标签
            nodes_with_labels = sorted(G.degree, key=lambda x: x[1], reverse=True)[:50]
            labels = {node: str(node) for node, _ in nodes_with_labels}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_family='sans-serif',
                                    verticalalignment='baseline')
        else:
            nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif', verticalalignment='baseline')

        # 如果边不太多，显示边上的interactions
        if G.number_of_edges() < 100:  # 减少显示标签的边数，因为交互列表可能很长
            print("Drawing interactions on edges...", end='')
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
                        label += f"... +{len(interactions) - 3} more"
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
        print(f"Visualization completed. Time elapsed: {end_time - start_time:.6f} seconds")

    def track_provenance(self, policy='lrb', target_node=None, target_timestamp=None):
        """
        追踪数据来源问题 (Provenance Problem)
        
        给定一个TIN G(V, E, R)，在任意时刻t和任意节点v∈V，确定缓冲区Bv中累积的总量的来源O(t, Bv)。
        O(t, Bv)是一组(τ,o,τ.q)三元组τ，其中每个数量τ.q由节点τ.o在时间τ.t生成，
        且满足∑(τ∈O(t,Bv)) τ.q = |Bv|。
        
        参数:
            policy: 追踪策略，'lrb'表示最早生成优先，'mrb'表示最新生成优先
            target_node: 目标节点，如果为None则追踪所有节点
            target_timestamp: 目标时间戳，如果指定则只考虑截至该时间的交互
            
        返回:
            节点的数据量缓冲区字典，每个节点对应一个包含(origin, time, quantity)三元组的列表
        """
        policy = policy.lower()
        if policy not in PROVENANCE_POLICIES:
            raise ValueError(f"策略必须为 {PROVENANCE_POLICIES} 中的一个")

        print(f"Tracking provenance using {'Least' if policy == 'lrb' else 'Most'}-recently born selection...")
        start_time = t.time()

        # 按时间排序所有交互
        sorted_interactions = sorted(self.interactions, key=lambda x: x[2])

        # 如果指定了目标时间戳，只考虑截至该时间的交互
        if target_timestamp is not None:
            sorted_interactions = [inter for inter in sorted_interactions if inter[2] <= target_timestamp]
            print(f"Limiting to interactions before or at timestamp {target_timestamp}")
            print(f"Found {len(sorted_interactions)} interactions within the time limit")

        # 初始化每个节点的缓冲区
        buffers = {node: [] for node in self.nodes}

        # 处理所有交互
        for i, (source, dest, time, quantity) in enumerate(sorted_interactions):
            if i % 100000 == 0 and i > 0:
                print(f"Processed {i} interactions...")

            # 剩余需要转移的数据量
            remaining_quantity = quantity

            # 当有剩余数据量且源节点缓冲区非空时，继续处理
            while remaining_quantity > 0 and buffers[source]:
                # 根据策略选择三元组
                if policy == 'lrb':
                    # LRB: 选择最早生成的三元组（最小堆按时间排序）
                    triple = buffers[source][0]
                    birth_time, origin, avail_quantity = triple

                    if avail_quantity > remaining_quantity:
                        # 分割三元组
                        heapq.heapreplace(buffers[source], (birth_time, origin, avail_quantity - remaining_quantity))
                        # 添加到目标节点
                        heapq.heappush(buffers[dest], (birth_time, origin, remaining_quantity))
                        remaining_quantity = 0
                    else:
                        # 移除整个三元组
                        heapq.heappop(buffers[source])
                        # 添加到目标节点
                        heapq.heappush(buffers[dest], (birth_time, origin, avail_quantity))
                        remaining_quantity -= avail_quantity
                else:
                    # MRB: 选择最近生成的三元组（使用最大堆，存储负时间）
                    triple = buffers[source][0]
                    neg_birth_time, origin, avail_quantity = triple

                    if avail_quantity > remaining_quantity:
                        # 分割三元组
                        heapq.heapreplace(buffers[source],
                                          (neg_birth_time, origin, avail_quantity - remaining_quantity))
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
                if policy == 'lrb':
                    heapq.heappush(buffers[dest], (time, source, remaining_quantity))
                else:  # MRB
                    heapq.heappush(buffers[dest], (-time, source, remaining_quantity))  # 注意负时间

        end_time = t.time()
        print(f"Provenance track complete, time elapsed: {end_time - start_time:.6f} seconds")

        # 如果指定了目标时间戳，添加到输出信息中
        time_info = f" at timestamp {target_timestamp}" if target_timestamp is not None else ""

        # 如果是MRB策略，需要转换时间为正值
        if policy == 'mrb':
            converted_buffers = {}
            # 如果指定了目标节点
            if target_node is not None:
                if target_node in buffers:
                    result = [(-neg_time, origin, quantity) for neg_time, origin, quantity in buffers[target_node]]
                    total_quantity = sum(quantity for _, _, quantity in result)
                    print(
                        f"Node {target_node}'s buffer{time_info} contains {len(result)} triples, total quantity: {total_quantity}")
                    return result
                else:
                    print(f"Node {target_node} does not exist")
                    return []

            # 转换所有节点的结果
            for node, buffer in buffers.items():
                converted_buffers[node] = [(-neg_time, origin, quantity) for neg_time, origin, quantity in buffer]
            return converted_buffers

        # 如果是LRB策略，直接返回结果
        if target_node is not None:
            if target_node in buffers:
                result = buffers[target_node]
                total_quantity = sum(quantity for _, _, quantity in result)
                print(
                    f"Node {target_node}'s buffer{time_info} contains {len(result)} triples, total quantity: {total_quantity}")
                return result
            else:
                print(f"Node {target_node} does not exist")
                return []

        return buffers

    def analyze_provenance(self, buffer):
        """
        分析追踪结果，返回数据来源统计
        
        参数:
            buffer: 缓冲区（可以是单个节点的三元组列表或多个节点的字典）
            
        返回:
            (origin_stats, buffer_str): 节点及其贡献的字典和缓冲区内容的字符串表示
        """
        # 检查输入类型，处理不同形式的buffer
        if isinstance(buffer, dict):
            # 多节点情况: buffer是{node: [(time, origin, quantity), ...], ...}形式
            if not buffer:
                return {}, ""

            buffer_str = ''
            for n, bf in buffer.items():
                # 生成当前节点的缓冲区字符串表示
                provenance_str = []
                for timestamp, origin, quantity in bf:
                    provenance_str.append(f"(o={origin}, ts={timestamp}, qty={quantity:.2f})")

                buffer_str += f"Buffer of vertex {n}:" + ' '.join(provenance_str) + '\n'

            return None, buffer_str

        else:
            # 单节点情况: buffer是[(time, origin, quantity), ...]形式
            if not buffer:
                return {}, ""

            provenance_str = []
            # 统计每个来源的贡献量
            origin_stats = defaultdict(float)
            for timestamp, origin, quantity in buffer:
                origin_stats[origin] += quantity
                provenance_str.append(f"(o={origin}, ts={timestamp}, qty={quantity:.2f})")

            # 按贡献量降序排序
            sorted_stats = sorted(origin_stats.items(), key=lambda x: x[1], reverse=True)

            return dict(sorted_stats), ' '.join(provenance_str)


if __name__ == "__main__":
    # 创建TIN实例
    tin = TemporalInteractionNetwork()
    choice = input("Which data do you want to use?\n1:test file (graph.txt) <DEFAULT>\n2:part of the taxi csv (tin_graph.txt)\n[(1)/2]=>")
    tin.create_from_taxi_data('tin_graph.txt' if choice and int(choice) == 2 else 'graph.txt')

    choice = input("\nDo you want to generate an image for the network?(Bad performance for massive graph)\n[Y/(n)]=>")
    if choice.lower() == 'y':
        # 可视化网络
        tin.visualize()

    # 询问用户选择数据追踪策略
    print("\nTrack provenance using which policy:")
    print("1. LRB (Least Recently Born) <DEFAULT>")
    print("2. MRB (Most Recently Born)")

    choice = input("(1) or 2: ").strip()
    idx = int(choice) if choice else 1
    policy = PROVENANCE_POLICIES[idx - 1]  # 修正索引从1开始

    # 输出统计信息
    stats = tin.get_statistics()
    print("\nNetwork statistics:")
    for key, value in stats.items():
        print(f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")

    # A. 询问用户要查询哪个节点
    print("\nWhich node to track provenance for?")
    print("Enter node ID or leave blank for all selection:")
    node_input = input("Node ID: ").strip()

    if node_input:
        try:
            target_node = int(node_input)
            # 检查节点是否存在
            if target_node not in tin.nodes:
                print(f"Node {target_node} not found. Using random selection.")
                target_node = random.choice(list(tin.nodes))
        except ValueError:
            print("Invalid input. Using random selection.")
            target_node = random.choice(list(tin.nodes))
    else:
        target_node = None

    print(f"Selected target node: {target_node if target_node else 'ALL'}")

    # B. 询问用户查询的时间戳
    print("\nTrack provenance up to which timestamp?")
    print("Enter timestamp or leave blank to consider all interactions:")
    timestamp_input = input("Timestamp: ").strip()

    target_timestamp = None
    if timestamp_input:
        try:
            target_timestamp = int(timestamp_input)
            # 验证时间戳在有效范围内
            all_times = [inter[2] for inter in tin.interactions]
            if all_times:
                min_time, max_time = min(all_times), max(all_times)
                if target_timestamp < min_time:
                    print(f"Warning: Timestamp {target_timestamp} is before earliest interaction ({min_time}).")
                elif target_timestamp > max_time:
                    print(f"Warning: Timestamp {target_timestamp} is after latest interaction ({max_time}).")
        except ValueError:
            print("Invalid timestamp. Considering all interactions.")
            target_timestamp = None

    # 使用选定策略追踪数据来源
    print(f"\nUsing {policy.upper()} strategy to track provenance...")
    provenance_buffer = tin.track_provenance(policy=policy, target_node=target_node, target_timestamp=target_timestamp)
    provenance_stats, buffer_string = tin.analyze_provenance(provenance_buffer)

    # 显示结果
    print(f"\n{policy.upper()} strategy sources for vertex {target_node if target_node else 'ALL'}:")
    if provenance_stats:
        for i, (origin, quantity) in enumerate(sorted(provenance_stats.items(), key=lambda x: x[1], reverse=True)[:10]):
            print(f"{i + 1}. Node {origin}: {quantity:.2f} units")
        print(f"Buffer of vertex {target_node}:" + buffer_string)
    else:
        # 显示缓冲区内容
        print(buffer_string)

    print("\nDone!")
