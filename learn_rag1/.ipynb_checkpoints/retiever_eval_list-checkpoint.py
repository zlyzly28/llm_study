import math


def calucate_dcg(res_score):
    dcg = 0
    for i in range(len(res_score)):
        # print(math.log2(i+2))
        dcg += (res_score[i] / math.log2(i + 2))
    return dcg


def calucate_idcg(res_score):
    idcg = 0
    sorted_res_score = sorted(res_score, reverse=True)
    for i in range(len(sorted_res_score)):
        # print(math.log2(i+2))
        idcg += (sorted_res_score[i] / math.log2(i + 2))
    return idcg


def ndcg_score_cal(lab_index, res, top_k=8, so=3):
    list1 = range(lab_index - so, lab_index + so + 1)
    n = so + 1
    lab_score = list(range(n)) + list(range(n - 2, -1, -1)) if n > 1 else []
    res_score = [0] * len(res)
    for index, ret_res in enumerate(res):
        if ret_res in list1:
            res_score[index] = lab_score[list1.index(ret_res)]
    dcg = calucate_dcg(res_score)
    idcg = calucate_idcg(res_score)
    if idcg == 0.0:
        return 0.0
    else:
        return dcg / idcg

def get_ndcg_score(lab_index_list, retriever_list):
    ans = 0
    for index, num in enumerate(lab_index_list):
        # print(num, retriever_list[index])
        ans += ndcg_score_cal(num, retriever_list[index])
    return ans / len(lab_index_list)


def soft_ht_score(lab_index, retriever_res, top_k = 8, so = 3)->float:
    list1 = range(lab_index-so, lab_index+so)
    if any(element in retriever_res[:top_k] for element in list1):
        return 1.0
    else:
        return 0.0

def soft_mmr_score(lab_index, retriever_res, top_k = 8, so = 3)->float:
    list1 = range(lab_index-so, lab_index+so)
    ank = 0
    for element in list1:
        if element in retriever_res[:top_k]:
            ank += 1.0 / (retriever_res.index(element) + 1)
    return ank

def ht_score(lab_index, retriever_res, top_k = 8, so = 3)->float:
    if lab_index in retriever_res[:top_k]:
        return 1.0
    else:
        return 0.0

def mmr_score(lab_index, retriever_res, top_k = 8, so = 3)->float:
    ank = 0
    if lab_index in retriever_res[:top_k]:
        ank += 1.0 / (retriever_res[:top_k].index(lab_index) + 1)
    return ank

def get_ht_score(lab_index_list, retriever_list):
    ans = 0
    for index, num in enumerate(lab_index_list):
        ans += ht_score(num, retriever_list[index])
    return ans / len(lab_index_list)

def get_mmr_score(lab_index_list, retriever_list):
    ans = 0
    for index, num in enumerate(lab_index_list):
        ans += mmr_score(num, retriever_list[index])
    return ans / len(lab_index_list)

def get_soft_ht_score(lab_index_list, retriever_list):
    ans = 0
    for index, num in enumerate(lab_index_list):
        ans += soft_ht_score(num, retriever_list[index])
    return ans / len(lab_index_list)

def get_soft_mmr_score(lab_index_list, retriever_list):
    ans = 0
    for index, num in enumerate(lab_index_list):
        ans += soft_mmr_score(num, retriever_list[index])
    return ans / len(lab_index_list)

def get_retriever_res_list(document_list, top_k = 20):
    id_tmp = []
    for n in range(len(document_list)):
        result = []
        for doc in document_list[n][:top_k]:
            result.append(doc.metadata['id'])
        id_tmp.append(result)
    return id_tmp
