import pandas as pd
from elasticsearch import Elasticsearch
from datasets import load_from_disk
import datetime
from pytz import timezone
import sys
import csv

# Global settings
BM25_PARAMETERS = (float(sys.argv[1]), float(sys.argv[2]))  # b and k1 for BM25
ES_HOST = "http://127.0.0.1:9200"  # Elasticsearch host address
INDEX_SETTINGS = {  # Elasticsearch index settings
    "settings": {
        "index": {
            "similarity": {
                "default": {
                    "type": "BM25",
                    "b": BM25_PARAMETERS[0],
                    "k1": BM25_PARAMETERS[1]
                }
            }
        }
    }
}
DATASET_PATH = "../data/train_dataset/"  # Path to the dataset
RESULTS_FILE = 'bm25_results.csv'  # File to store the results


class ElasticRetrieval:
    def __init__(self, es, index_name):
        self.es = es
        self.index_name = index_name

    def search(self, query, topk=30):
        body = {
            'query': {
                'match': {
                    'content': {
                        'query': query,
                        'fuzziness': 'AUTO'
                    }
                }
            }
        }
        res = self.es.search(index=self.index_name, body=body, size=topk)
        return res['hits']['hits']


def count_positions(dataset):
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


def main():
    es = Elasticsearch([ES_HOST])
    index_name = f'odqa_dataset_by_bm25_b{BM25_PARAMETERS[0]}_k1{BM25_PARAMETERS[1]}_{datetime.datetime.now(timezone("Asia/Seoul")).strftime("%m%d%H%M%S")}'
    es.indices.create(index=index_name, body=INDEX_SETTINGS)

    while True:
        index_health = es.cluster.health(index_name)
        if index_health['status'] in ['yellow', 'green']:
            break

    print(f"b: {BM25_PARAMETERS[0]}, k1: {BM25_PARAMETERS[1]}")
    print(es.indices.get_settings(index=index_name)[index_name]['settings']['index']['similarity']['default'])

    dataset = load_from_disk(DATASET_PATH)
    for i, data in enumerate(dataset['train']):
        es.index(index=index_name, id=i, body={'content': data['context']})

    retriever = ElasticRetrieval(es, index_name)
    dataset = pd.DataFrame(dataset['validation'])

    relevance_positions, bm25_scores = [], []
    for _, row in dataset.iterrows():
        results = retriever.search(row['question'])
        found_position = -1
        found_score = None
        for i, result in enumerate(results):
            if row['context'] == result['_source']['content']:
                found_position = i
                found_score = result['_score']
                break
        relevance_positions.append(found_position)
        bm25_scores.append(found_score)

    dataset['relevance_position'] = relevance_positions
    dataset['bm25_score'] = bm25_scores

    count_dict = count_positions(dataset)
    print(count_dict)

    with open(RESULTS_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([BM25_PARAMETERS[0], BM25_PARAMETERS[1], dataset['bm25_score'].mean(), count_dict])


if __name__ == "__main__":
    main()

