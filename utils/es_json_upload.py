from elasticsearch import Elasticsearch
from datasets import load_from_disk
import pickle
from transformers import BertTokenizer


dataset = load_from_disk("../data/train_dataset/")
es = Elasticsearch(["http://127.0.0.1:9200"])  # ElasticSearch 서버의 주소를 입력해주세요. 예: Elasticsearch(['http://localhost:9200/'])


# BM25 모델을 사용하기 위한 Elasticsearch의 설정값입니다.
settings = {
    "settings": {
        "index": {
            "similarity": {
                "default": {
                    "type": "BM25",
                    "b": 0.75,
                    "k1": 1.2
                }
            }
        }
    }
}



# 1) context 데이터를 바로 Elasticsearch에 저장
INDEX_NAME = 'odqa_dataset_by_bm25_yunjin' # es 에 올릴 인덱스 이름을 지정 (겹치면 에러 발생)
es.indices.create(index=INDEX_NAME, body=settings) # 인덱스 생성 (= DB 스키마 생성)

for i, data in enumerate(dataset['train']):
    es.index(index=INDEX_NAME, id=i, body={'content': data['context']})



# <토크나이징 해서 es에 업로드 하고 검색 것은 추후 수정에서 올리겠습니다>
# # 2) 토크나이징된 데이터를 Elasticsearch에 저장하는 코드
# INDEX_NAME = 'odqa_dataset_by_bm25_tokenizing' # es 에 올릴 인덱스 이름을 지정 (겹치면 에러 발생)
# es.indices.create(index=INDEX_NAME, body=settings) # 인덱스 생성 (= DB 스키마 생성)
#
# tokenizer = BertTokenizer.from_pretrained('monologg/kobert')  # KoBERT tokenizer
# for i, data in enumerate(dataset['train']):
#     tokenized_content = tokenizer.tokenize(data['context'])  # KoBERT로 토크나이징
#     restored_content = ' '.join(tokenizer.convert_tokens_to_string(token) for token in tokenized_content)  # 토큰을 문자열로 변환
#     es.index(index=INDEX_NAME+"tokenizing", id=i, body={'original_content': data['context'], 'tokenized_content': restored_content})  # 원문과 변환한 문자열을 Elasticsearch에 저장