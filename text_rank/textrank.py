# -*- coding: utf-8 -*-

"""
@Time    : 2018/10/31 21:32
@Author  : fazhanzhang
@Function :
"""

import re
import jieba
import numpy as np
from nltk.cluster.util import cosine_distance


MIN_SEQ_LEN = 5
STOPWORDS_PATH = './stopwords.txt'


def load_stopwords(file_path):
    with open(file_path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def split_doc(doc, stopwords=None):
    if not stopwords:
        stopwords = []

    sentences = []
    cut_sentences = []
    origin_sentences = []

    while len(doc) > 0:
        for i in range(len(doc)):
            if doc[i] in ['。', '！', '?', '？']:
                sentences.append(doc[:i+1])
                doc = doc[i+1:]
                break
    for sent in sentences:
        if len(sent) > MIN_SEQ_LEN:
            cut_sentences.append([word for word in jieba.lcut(sent) if word not in stopwords])
            origin_sentences.append(sent)
    return origin_sentences, cut_sentences


def sentence_similarity(sent1, sent2):
    """
    计算两个句子之间的相似性
    :param sent1:
    :param sent2:
    :return:
    """
    all_words = list(set(sent1 + sent2))

    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)

    for word in sent1:
        vector1[all_words.index(word)] += 1

    for word in sent2:
        vector2[all_words.index(word)] += 1

    # cosine_distance 越大越不相似
    return 1-cosine_distance(vector1, vector2)


def build_similarity_matrix(sentences):
    """
    构建相似矩阵
    :param sentences:
    :return:
    """
    S = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            S[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2])
    # 将矩阵正则化
    for idx in range(len(S)):
        if S[idx].sum == 0:
            continue
        S[idx] /= S[idx].sum()

    return S


def pagerank(A, eps=0.0001, d=0.85):
    P = np.ones(len(A)) / len(A)
    while True:
        new_P = np.ones(len(A)) * (1 - d) / len(A) + d * A.T.dot(P)
        delta = abs(new_P - P).sum()
        if delta <= eps:
            return new_P
        P = new_P


def textrank(doc, ratio=0.2):
    stopwords = load_stopwords(STOPWORDS_PATH)

    origin_sentences, cut_sentences = split_doc(doc, stopwords=stopwords)

    S = build_similarity_matrix(cut_sentences)

    sentences_ranks = pagerank(S)

    sentences_ranks = [item[0] for item in sorted(enumerate(sentences_ranks), key=lambda item: -item[1])]

    selected_sentences_index = sorted(sentences_ranks[:int(len(origin_sentences)*ratio)])

    summary = []
    for idx in selected_sentences_index:
        summary.append(origin_sentences[idx])

    return ''.join(summary)


if __name__ == '__main__':
    doc = """习近平在主持学习时发表了讲话。他强调，人工智能是引领这一轮科技革命和产业变革的战略性技术，具有溢出带动性很强的“头雁”效应。在移动互联网、大数据、超级计算、传感网、脑科学等新理论新技术的驱动下，人工智能加速发展，呈现出深度学习、跨界融合、人机协同、群智开放、自主操控等新特征，正在对经济发展、社会进步、国际政治经济格局等方面产生重大而深远的影响。加快发展新一代人工智能是我们赢得全球科技竞争主动权的重要战略抓手，是推动我国科技跨越发展、产业优化升级、生产力整体跃升的重要战略资源。

　　习近平指出，人工智能具有多学科综合、高度复杂的特征。我们必须加强研判，统筹谋划，协同创新，稳步推进，把增强原创能力作为重点，以关键核心技术为主攻方向，夯实新一代人工智能发展的基础。要加强基础理论研究，支持科学家勇闯人工智能科技前沿的“无人区”，努力在人工智能发展方向和理论、方法、工具、系统等方面取得变革性、颠覆性突破，确保我国在人工智能这个重要领域的理论研究走在前面、关键核心技术占领制高点。要主攻关键核心技术，以问题为导向，全面增强人工智能科技创新能力，加快建立新一代人工智能关键共性技术体系，在短板上抓紧布局，确保人工智能关键核心技术牢牢掌握在自己手里。要强化科技应用开发，紧紧围绕经济社会发展需求，充分发挥我国海量数据和巨大市场应用规模优势，坚持需求导向、市场倒逼的科技发展路径，积极培育人工智能创新产品和服务，推进人工智能技术产业化，形成科技创新和产业应用互相促进的良好发展局面。要加强人才队伍建设，以更大的决心、更有力的措施，打造多种形式的高层次人才培养平台，加强后备人才培养力度，为科技和产业发展提供更加充分的人才支撑。"""
    doc = re.sub("\s", "", doc)

    summary = textrank(doc)
    print(doc)
    print(summary)