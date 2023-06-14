import pandas as pd
from elasticsearch import Elasticsearch
from transformers import BertTokenizer
from typing import List
import pandas as pd
from datasets import load_from_disk

dataset = load_from_disk("../data/train_dataset/")['validation']

tokenizer = BertTokenizer.from_pretrained('monologg/kobert')  # KoBERT tokenizer


# Elasticsearch 에서 데이터를 검색하는 클래스
class ElasticRetrieval:
    def __init__(self, index_name):
        self.es = Elasticsearch()
        self.index_name = index_name

    # es 에서 데이터를 검색하는 함수(context 그대로 올라간 경우)
    def search(self, query, topk=30):
        body = {
            'query': {
                'match': {
                    'content': { # <주의> content 로 저장했기 때문에 'content' 컬럼에 대해서 검색하는 것!!
                        'query': query,
                        'fuzziness': 'AUTO'
                    }
                }
            }
        }
        res = self.es.search(index=self.index_name, body=body, size=topk)
        return res['hits']['hits']

# ground truth 를 리스트로 반환
def get_relevance_positions(dataset: pd.DataFrame, retriever: ElasticRetrieval) -> List[int]:
    relevance_positions = []

    for i, row in dataset.iterrows():
        results = retriever.search(row['question'])
        found_position = -1  # If ground truth is not found in the results, -1 will be stored.

        for j, result in enumerate(results):
            if row['context'] == result['_source']['content']:
                found_position = j
                break

        relevance_positions.append(found_position)

    return relevance_positions


# es 에서 가져온 결과를 기반으로 정답이 몇 번째에 있는지 확인하는 함수
def valid_es_bm25(dataset: pd.DataFrame):
    count_dict = {"in5": 0, "in10": 0, "in15": 0, "in20": 0, "in25": 0, "in30": 0, "out": 0}

    for d in dataset['relevance_position']:
        if d < 0:
            count_dict["out"] += 1
        elif d < 5:
            count_dict["in5"] += 1
        elif d < 10:
            count_dict["in10"] += 1
        elif d < 15:
            count_dict["in15"] += 1
        elif d < 20:
            count_dict["in20"] += 1
        elif d < 25:
            count_dict["in25"] += 1
        elif d < 30:
            count_dict["in30"] += 1

    return count_dict



# 1)
INDEX_NAME = 'odqa_dataset_by_bm25' # es_sjon_upload 에서 지정했던 index 이름
retriever = ElasticRetrieval(INDEX_NAME)

dataset = pd.DataFrame(dataset)
relevance_positions = get_relevance_positions(dataset, retriever) # ground truth 위치 찾기

# Add the positions to the dataset
dataset['relevance_position'] = relevance_positions
print("1) es search by bm25")
print(valid_es_bm25(dataset))

