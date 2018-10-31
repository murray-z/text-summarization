# -*- coding: utf-8 -*-

"""
基于MMR实现自动文本摘要
参考：http://www.cs.cmu.edu/afs/.cs.cmu.edu/Web/People/jgc/publication/MMR_DiversityBased_Reranking_SIGIR_1998.pdf
"""

import jieba
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


STOPWORDS_PATH = './stopwords.txt'


def load_stopwords(stopwords_path):
    """
    加载停用词词典
    :param stopwords_path:
    :return:
    """
    with open(stopwords_path, encoding='utf-8') as f:
        return [line.strip() for line in f]


def clean_sentence(sentence, stopwords):
    """
    过滤句子中的停用词
    :param sentence:
    :param stopwords:
    :return:
    """
    return " ".join([word for word in sentence if word not in stopwords])


def calculate_similarity(sentence, doc):
    """
    计算余弦相似度
    :param sentence:
    :param doc:
    :return:
    """
    if doc == []:
        return 0

    doc_str = ' '.join(doc)
    vocab = list(set((sentence + ' ' + doc_str).split()))

    cv = CountVectorizer(vocabulary=vocab)

    docVector = cv.fit_transform([doc_str])
    sentenceVector = cv.fit_transform([sentence])

    return cosine_similarity(docVector, sentenceVector)[0][0]


def clean_data(doc, stopwords):
    """
    清洗输入文本
    :param doc:
    :param stopwords:
    :return:
    """
    separators = ["。", "？", "!", "！", "?"]
    for sep in separators:
        doc = doc.replace(sep, sep+"##")

    sentences = doc.split("##")[:-1]

    clean_sentences = []
    origin_clean_dic = {}
    origin_sent_order = {}
    for idx, sent in enumerate(sentences):
        clean_sent = clean_sentence(jieba.lcut(sent), stopwords=stopwords)
        clean_sentences.append(clean_sent)
        origin_clean_dic[clean_sent] = sent
        origin_sent_order[sent] = idx

    return clean_sentences, origin_clean_dic, origin_sent_order


def calculate_scores_sentence_to_doc(clean_sentences):
    """
    计算每个句子和文本之间的相似性
    :param clean_sentences:
    :return:
    """
    scores = {}
    set_clean_sentences = set(clean_sentences)
    for sentence in clean_sentences:
        temp_doc = set_clean_sentences - set([sentence])
        score = calculate_similarity(sentence, list(temp_doc))
        scores[sentence] = score
    return scores


def mmr(doc, alpha=0.5, ratio=0.2):
    """
    主流程
    :param doc:
    :param alpha:
    :param ratio:
    :return:
    """
    stopwords = load_stopwords(STOPWORDS_PATH)

    clean_sentences, origin_clean_dic, origin_sent_order = clean_data(doc, stopwords=stopwords)

    scores = calculate_scores_sentence_to_doc(clean_sentences)

    summary_num = len(clean_sentences)*ratio

    summary = {}

    while summary_num > 0:
        mmr = {}
        for sentence in scores:
            if not sentence in summary:
                mmr[sentence] = alpha * scores[sentence] - (1-alpha) * calculate_similarity(sentence, summary)
        selected = sorted(mmr.items(), key=lambda item: -item[1])[0][0]
        summary[origin_clean_dic[selected]] = origin_sent_order[origin_clean_dic[selected]]
        summary_num -= 1

    summary = "".join([sent for (sent, order) in sorted(summary.items(), key=lambda item: item[1])])
    return summary


if __name__ == "__main__":
    doc = """习近平在主持学习时发表了讲话。他强调，人工智能是引领这一轮科技革命和产业变革的战略性技术，具有溢出带动性很强的“头雁”效应。在移动互联网、大数据、超级计算、传感网、脑科学等新理论新技术的驱动下，人工智能加速发展，呈现出深度学习、跨界融合、人机协同、群智开放、自主操控等新特征，正在对经济发展、社会进步、国际政治经济格局等方面产生重大而深远的影响。加快发展新一代人工智能是我们赢得全球科技竞争主动权的重要战略抓手，是推动我国科技跨越发展、产业优化升级、生产力整体跃升的重要战略资源。

    　　习近平指出，人工智能具有多学科综合、高度复杂的特征。我们必须加强研判，统筹谋划，协同创新，稳步推进，把增强原创能力作为重点，以关键核心技术为主攻方向，夯实新一代人工智能发展的基础。要加强基础理论研究，支持科学家勇闯人工智能科技前沿的“无人区”，努力在人工智能发展方向和理论、方法、工具、系统等方面取得变革性、颠覆性突破，确保我国在人工智能这个重要领域的理论研究走在前面、关键核心技术占领制高点。要主攻关键核心技术，以问题为导向，全面增强人工智能科技创新能力，加快建立新一代人工智能关键共性技术体系，在短板上抓紧布局，确保人工智能关键核心技术牢牢掌握在自己手里。要强化科技应用开发，紧紧围绕经济社会发展需求，充分发挥我国海量数据和巨大市场应用规模优势，坚持需求导向、市场倒逼的科技发展路径，积极培育人工智能创新产品和服务，推进人工智能技术产业化，形成科技创新和产业应用互相促进的良好发展局面。要加强人才队伍建设，以更大的决心、更有力的措施，打造多种形式的高层次人才培养平台，加强后备人才培养力度，为科技和产业发展提供更加充分的人才支撑。"""

    doc = re.sub("\s", "", doc)
    summary = mmr(doc)

    print(doc)
    print(summary)





