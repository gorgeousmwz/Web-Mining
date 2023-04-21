# date : 2023-4-18
# language : python
# version : 3.7.16
# coder : mwz
# email : mawenzhuogorgeous@gmail.com
# topic : web mining
# details : The author network and author unit geographic network are constructed respectively according to the Authors or Addresses field.
#           Combine network analysis and other field information to find the most influential person (unit), the main innovator (unit), the main communicator (unit);
#           Divide researchers or units into groups and visualize them
# relative link: https://zhuanlan.zhihu.com/p/86004363 -> PageRank

import os
import pandas as pd
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import prettytable as pt


def readFile(path,constraint=None,no_values=''):
    '''
    读取数据文件,返回list
    params:
        path: 数据文件夹路径
        constraint: 读取指定列的列名称列表
        no_values: 空缺值处理
    returns:
        data_sum:数据列表
    '''
    files=os.listdir(path)
    data_sum=[]
    for filename in files:
        file=os.path.join(path,filename)
        data=pd.read_excel(io=file,usecols=constraint) # 读取文件
        data.fillna(no_values,inplace=True) # 填写空缺值
        data=data.values.tolist() # 转为list
        data_sum+=data
    return data_sum

def dataPrepoccess(data):
    '''
    数据预处理:将字段中的作者/单位全部分割,组成database
    '''
    database=[]
    for i in range(len(data)):
        database.append([])
        a=data[i][0].split('; ') # 一定要注意分割符号是';'+space
        if a!=['']:
            database[i]+=a
    return database

def dataStatistics(data):
    '''
    数据统计，统计每个关键词对应的作者和发表年份
    '''
    database={}
    for i in range(len(data)):
        a=data[i][0].split('; ')[0] # First Author
        b=data[i][1].split('; ') # Author Keywords
        c=data[i][2].split('; ') # Keywords Plus
        d=2025
        if data[i][3] != '':
            d=int(data[i][3]) # Publication Year
        keywords=b+c
        if keywords !=['']:
            for keyword in keywords:
                if database.get(keyword.lower(),(a,d))[1]>= d: # 如果这个关键词在本篇文章中发表年份更小，则替换作者和年份
                    database[keyword.lower()]=(a,d)
    return database

def get_subsets(lst, k):
    """
    将一个列表转换为set,获取其所有长度为k的子集并转换为元组列表返回
    """
    s = set(lst)
    subsets = list(itertools.combinations(s, k))
    return [tuple(sorted(subset)) for subset in subsets]#[一定要sorted啊，不然每次找出来的边数目不一样]

def getEdgeAndWeights(database,constraint=1):
    '''
    获取边和权重
    在这个问题中,边权重为边两个端点所代表的作者/单位合作的次数
    params:
        database: 数据库
        constraint: 权重阈值,只考权重大于>=constraint的边
    returns:
        edges: 边和权重列表
    '''
    edges_wights={}
    for paper in database: # 遍历数据库中每篇文章
        cooperations=get_subsets(paper,2) # 得到一篇文章中两个作者之间的合作列表
        for cooperation in cooperations: # 遍历每个合作
            edges_wights[cooperation]=edges_wights.get(cooperation,0)+1 # 计数为一个edge，如果已经存在则权重+1
    edges=[]
    for edge,weight in edges_wights.items(): # 遍历字典
        if weight>=constraint: # 如果大于权重阈值
            edges.append(tuple(list(edge)+[weight])) # 格式转换
    return edges

def getNodes(edges):
    '''
    获取节点列表(每个节点就是一个作者/单位)
    params:
        edges: 边-权重列表
    returns:
        nodes: 字典升序节点列表
    '''
    nodes=[]
    for node1,node2,_ in edges: # 遍历边，添加端点
        if node1 not in nodes:
            nodes.append(node1)
        if node2 not in nodes:
            nodes.append(node2)
    nodes.sort() # 字典升序的节点列表
    return nodes

def createNet(nodes,edges,save_path):
    '''
    创建网络,并保存网络
    params:
        nodes: 节点列表
        edges: 边-权重字典
        save_path: 网络保存地址
    returns:
        net: 网络
    '''
    net=nx.Graph() # 创建空的无向图
    net.add_nodes_from(nodes)# 添加节点
    net.add_weighted_edges_from(edges) # 添加边-权重
    # 数据展示
    print('#网络结构表单:(仅展示权值最大的10条边)')
    data=[]
    sorted_edges=sorted(list(net.edges(data=True)),reverse=True,key=lambda x:x[2]['weight'])
    for edge in sorted_edges[0:10]:
        data.append([edge[0],edge[1],net.get_edge_data(edge[0],edge[1])['weight']])
    prettyPrint(['Node1','Node2','Weight'],data)
    nx.write_gexf(net, save_path) # 保存网络
    return net

def drawNet(net,nodes,edges,path):
    '''
    绘制网络图
    节点度数越大：蓝色->紫色
    边权重越大：黄色->橙色
    '''
    node_size=[25]*len(nodes)
    node_color = [nx.degree(net,node) for node in nodes] # 节点的度数为其颜色的深浅程度
    edge_color = [net.get_edge_data(node1,node2)['weight'] for node1,node2,_ in edges] # 边的权重为其颜色的
    # 绘制网络
    pos = nx.circular_layout(net)  # 布局方式
    nx.draw_networkx_nodes(net, pos, node_size=node_size, node_color=node_color, cmap=plt.cm.cool)
    nx.draw_networkx_labels(net, pos, font_size=6, font_color='black', font_family='sans-serif')
    nx.draw_networkx_edges(net, pos, edge_color=edge_color, width=1, edge_cmap=plt.cm.Wistia)
    plt.savefig(path,dpi=300) # 保存图片

def NetworkOverallEvaluation(net):
    '''
    网络整体评价
    params:
        net: 网络
    '''
    print(f'网络节点数: {net.number_of_nodes()}')
    print(f'网络边数: {net.number_of_edges()}')
    print(f'网络密度: {nx.density(net)}')
    print(f'网络是否为连通图: {nx.is_connected(net)}')

def getPageRank(net,d=0.85,end_threshold=1e-6):
    '''
    页面排名算法
    params:
        net: 网络
        d: 阻尼系数
        end_threshold: 迭代结束阈值
    returns:
        pr: 每个节点的PageRank值字典
        max_node: PageRank值最大的节点
        max_pr: 最大的pagerank中心性
    '''
    pr=nx.pagerank(net,weight='weight',alpha=d,tol=end_threshold) # 计算每个节点的PageRank值
    pr=sorted(pr.items(), key=lambda x:x[1], reverse=True) # 排序
    max_node=pr[0][0] # PageRank Centrality最大的节点
    max_pr=pr[0][1] # PageRank Centrality最大值
    # 数据展示
    print('#页面排名中心性表格:(仅展示前十名节点的页面排名中心性)')
    data=[]
    for item in pr[0:10]:
        data.append([item[0],item[1]])
    prettyPrint(['Author','PageRank Centrality'],data)
    return pr,max_node,max_pr

def getDegreeCentrality(net):
    '''
    计算点度中心性
    params:
        net: 网络
    returns:
        degree_centrality: 节点的点度中心性字典
        max_node: 最大点度中心性节点
        max_dc: 最大的点度中心性
    '''
    degree_centrality = nx.degree_centrality(net) # 计算点度中心性
    degree_centrality=sorted(degree_centrality.items(), key=lambda x:x[1], reverse=True) # 排序
    max_node=degree_centrality[0][0] # Degree Centrality最大的节点
    max_dc=degree_centrality[0][1] #  Degree Centrality最大值
    # 数据展示
    print('#点度中心性表格:(仅展示前十名节点的点度中心性)')
    data=[]
    for item in degree_centrality[0:10]:
        data.append([item[0],item[1]])
    prettyPrint(['Author','Degree Centrality'],data)
    return degree_centrality,max_node,max_dc

def getBetweennessCentrality(net):
    '''
    计算中间中心性
    params:
        net: 网络
    returns:
        betweenness_centrality: 节点的中间中心性字典
        max_node: 最大中间中心性节点
        max_bc: 最大的中间中心性
    '''
    betweenness_centrality = nx.betweenness_centrality(net) # 计算中间中心性
    betweenness_centrality=sorted(betweenness_centrality.items(), key=lambda x:x[1], reverse=True) # 排序
    max_node=betweenness_centrality[0][0] # Betweenness Centrality最大的节点
    max_bc=betweenness_centrality[0][1] # Betweenness Centrality最大值
    # 数据展示
    print('#中间中心性表格:(仅展示前十名节点的中间中心性)')
    data=[]
    for item in betweenness_centrality[0:10]:
        data.append([item[0],item[1]])
    prettyPrint(['Author','Betweenness Centrality'],data)
    return betweenness_centrality,max_node,max_bc

def getClosenessCentrality(net):
    '''
    计算接近中心性
    params:
        net: 网络
    returns:
        betweenness_centrality: 节点的接近中心性字典
        max_node: 最大接近中心性节点
        max_cc: 最大的接近中心性
    '''
    closeness_centrality = nx.closeness_centrality(net) # 计算中间中心性
    closeness_centrality=sorted(closeness_centrality.items(), key=lambda x:x[1], reverse=True) # 排序
    max_node=closeness_centrality[0][0] # Closeness Centrality最大的节点
    max_cc=closeness_centrality[0][1] # Closeness Centrality最大值
    # 数据展示
    print('#接近中心性表格:(仅展示前十名节点的接近中心性)')
    data=[]
    for item in closeness_centrality[0:10]:
        data.append([item[0],item[1]])
    prettyPrint(['Author','Closeness Centrality'],data)
    return closeness_centrality,max_node,max_cc

def getMainInnovator(data_floder):
    '''
    寻找主要创新者: 本问题中我们将一个关键词的最早出现的那篇文章作为该文章第一作者的一次创新,最后返回创新最多的作者
    params:
        data_floder: 数据文件夹
    returns:
        main_innovator: 最主要创新者
        innovate_count: 最主要创新者创新次数
    '''
    data=readFile(data_floder,['Authors','Author Keywords','Keywords Plus','Publication Year']) # 读取文件
    database=dataStatistics(data) # 数据分析
    Innovator={} # 作者创新表
    for keyword,earliest in database.items(): # 遍历每条记录
        author,year=earliest # 最早提出这个关键词的作者和年份
        Innovator[author]=Innovator.get(author,[0,[]])
        Innovator[author][0]+=1 # 创新次数+1
        Innovator[author][1]+=[(keyword,year)] # 记录该作者对应的创新关键词和年份
    Innovator=sorted(Innovator.items(), key=lambda x:x[1][0], reverse=True) # 排序
    main_innovator=Innovator[0][0] # 最主要创新者
    innovate_count=Innovator[0][1][0] # 最主要创新者创新次数
    # 数据展示
    print('#创新者表格:(仅展示前十名创新者以及它们的三次创新内容)')
    data=[]
    for item in Innovator[0:10]:
        data.append([item[0],item[1][0],item[1][1][0:3]])
    prettyPrint(['Author','Innovation Times','Innovation Detail'],data)
    return main_innovator,innovate_count
    
def prettyPrint(col_names,data):
    '''
    美化输出
    params:
        col_names: 表头名称列表
        data: 数据
    '''
    tb=pt.PrettyTable() # 创建表格
    tb.field_names=['index']+col_names # 设置表头
    index=0
    for item in data:
        index+=1
        tb.add_row([index]+item) # 添加行
    print(tb)

def web_mining(data_floder,object,weight_threshold=1):
    '''
    网络挖掘
    params:
        data_floder: 数据文件夹路径
        object: 研究对象列名称 ['Authors'] or ['Addresses']
        weight_threshold: 权重阈值（低于阈值的边不予考虑）
    '''
    print('【文件读取】')
    data=readFile(data_floder,object) # 读取文件
    print('【数据预处理】')
    database=dataPrepoccess(data) # 数据预处理
    print('【创建网络】')
    edges=getEdgeAndWeights(database,weight_threshold) # 获取边和权重列表
    nodes=getNodes(edges) # 获取节点列表
    net=createNet(nodes,edges,f'Spatio-temporal_data_mining_and_analysis/Web-Mining/net/network[{object[0]}]_wt{weight_threshold}_node{len(nodes)}_edge{len(edges)}.gexf') # 创建网络
    drawNet(net,nodes,edges,f'Spatio-temporal_data_mining_and_analysis/Web-Mining/image/network[{object[0]}]_wt{weight_threshold}_node{len(nodes)}_edge{len(edges)}.png') # 绘制网络
    print('【网络整体评价】')
    NetworkOverallEvaluation(net) # 网络整体评价
    print('【网络分析】')
    _,max_node1,max_pr=getPageRank(net) # PageRank算法寻找最有影响力的人
    _,max_node2,max_dc=getDegreeCentrality(net) # 点度中心性计算最有影响力的人
    _,max_node3,max_bc=getBetweennessCentrality(net) # 中间中心性计算主要传播者
    _,max_node4,max_cc=getClosenessCentrality(net) # 接近中心性计算主要传播者
    main_innovator,innovate_count=getMainInnovator(data_floder) # 根据创新次数计算主要创新者
    print('#分析结果如下:')
    print(f'[PageRank Centrality]影响力最大的人/单位: {max_node1} -> PageRank Centrality值: {max_pr}')
    print(f'[Degree Centrality]影响力最大的人/单位: {max_node2} -> Degree Centrality值: {max_dc}')
    print(f'[Betweenness Centrality]主要传播的人/单位: {max_node3} -> Betweenness Centrality值: {max_bc}')
    print(f'[Closeness Centrality]主要传播的人/单位: {max_node4} -> BCloseness Centrality值: {max_cc}')
    print(f'[Innovation Times]最主要创新者/单位: {main_innovator} -> 创新次数: {innovate_count}')
    
web_mining("Spatio-temporal_data_mining_and_analysis/data",["Authors"],10)
