# -*- coding: utf-8 -*-

"""
@Time    : 2018/10/31 20:54
@Author  : fazhanzhang
@Function :
@参考： https://nlpforhackers.io/textrank-text-summarization/
"""

'''
假设:
1. N个网页
2. P = PageRank vector (P[i] 第i个页面 pagerank )
3. A[i][j] 用户从页面i转向j的概率
4. A[i][j] 若i和j相连，初始值 = 1/(由页面i 指向其他页面的连接数目), 否则为0
5. 如果i没有和其他页面相连, A[i][j] = i/N
'''

'''
pagerank 两个重要参数：
eps: 当两次迭代之间的差距小于该值，则算法停止；
d: 阻尼因子，用户随机选择一个页面的概率为i-d
'''

import numpy as np


# 假设我们有以下页面
links = {
    'webpage-1': set(['webpage-2', 'webpage-4', 'webpage-5', 'webpage-6', 'webpage-8', 'webpage-9', 'webpage-10']),
    'webpage-2': set(['webpage-5', 'webpage-6']),
    'webpage-3': set(['webpage-10']),
    'webpage-4': set(['webpage-9']),
    'webpage-5': set(['webpage-2', 'webpage-4']),
    'webpage-6': set([]),  # dangling page
    'webpage-7': set(['webpage-1', 'webpage-3', 'webpage-4']),
    'webpage-8': set(['webpage-1']),
    'webpage-9': set(['webpage-1', 'webpage-2', 'webpage-3', 'webpage-8', 'webpage-10']),
    'webpage-10': set(['webpage-2', 'webpage-3', 'webpage-8', 'webpage-9']),
}


def build_index(links):
    """
    构建页面索引
    :param links:
    :return:
    """
    website_list = links.keys()
    return {website: index for index, website in enumerate(website_list)}


def build_transition_matrix(links, index):
    """
    构建转移矩阵
    :param links:
    :param index:
    :return:
    """
    total_links = 0
    A = np.zeros((len(index), len(index)))
    for webpage in links:
        # 没有指向其他页面
        if not links[webpage]:
            # 相同概率
            A[index[webpage]] = np.ones(len(index)) / len(index)
        else:
            for dest_webpage in links[webpage]:
                total_links += 1
                A[index[webpage]][index[dest_webpage]] = 1.0 / len(links[webpage])
    return A


def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1-d)/len(A) + d * A.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P


if __name__ == '__main__':
    website_index = build_index(links)
    print(website_index)

    A = build_transition_matrix(links, website_index)
    print(A)

    result = pagerank(A)
    print(result)

    print([item[0] for item in sorted(enumerate(result), key=lambda item: -item[1])])





