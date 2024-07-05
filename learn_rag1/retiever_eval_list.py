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
    if any(element in retriever_res for element in list1):
        return 1.0
    else:
        return 0.0

def soft_mmr_score(lab_index, retriever_res, top_k = 8, so = 3)->float:
    list1 = range(lab_index-so, lab_index+so)
    ank = 0
    for element in list1:
        if element in retriever_res:
            ank += 1.0 / (retriever_res.index(element) + 1)
    return ank

def ht_score(lab_index, retriever_res, top_k = 8, so = 3)->float:
    if lab_index in retriever_res:
        return 1.0
    else:
        return 0.0

def mmr_score(lab_index, retriever_res, top_k = 8, so = 3)->float:
    ank = 0
    if lab_index in retriever_res:
        ank += 1.0 / (retriever_res.index(lab_index) + 1)
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


def get_result_retrieva(col_id, r_result, topk = 10):
    rerank_evaldict = {}

    for i in range(topk):
        # print('top_', i+1)
        res_retriever_list = get_retriever_res_list(r_result, i+1)
        # print(res_retriever_list[0])
        ht_score = get_ht_score(col_id, res_retriever_list)
        mmr_score = get_mmr_score(col_id, res_retriever_list)
        # print('ht:', round(ht_score, 3))
        # print('mmr_score:', round(mmr_score, 3))
        soft__ht_score = get_soft_ht_score(col_id, res_retriever_list)
        soft__mmr_score = get_soft_mmr_score(col_id, res_retriever_list)
        # print('soft__ht_score:', round(soft__ht_score, 3))
        # print('soft__mmr_score:', round(soft__mmr_score, 3))
        ndcg_score = get_ndcg_score(col_id, res_retriever_list)
        # print('ndcg:', round(ndcg_score, 3))
        rerank_evaldict['top_' + str(i+1)] = {
            "ht_score": round(ht_score, 3),
            'mmr_score': round(mmr_score, 3),
            'soft_ht_score': round(soft__ht_score, 3),
            'soft_mmr_score': round(soft__mmr_score, 3),
            'ndcg': round(ndcg_score, 3),
        }
    return rerank_evaldict