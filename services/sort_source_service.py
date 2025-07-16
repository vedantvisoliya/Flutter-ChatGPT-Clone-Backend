from sentence_transformers import SentenceTransformer # type: ignore
import numpy as np # type: ignore

class SortSourceService:

    def __init__(self):
        self.embedding_model = SentenceTransformer("all-miniLM-L6-v2")

    # i am using COS SIMILARITY SEARCH

    # what is embedding or vectorisation?
    # -> in this case we have query which is converted into 
    # a high order or high dimension space matrix

    # in cos similarity search we check the angle between
    # query vector and result vector and check there similarity
    # that the content we got from the web and check is the content
    # a valid result for the query.

    # we take the tail of both vectors and compare theta
    # if angle between the both vector is less that means
    # the result can be valid for the query

    # in cos theta case lesser the theta greater the value
    # that means if theta is 0 degree i.e value will be 1 which imples the 
    # vectors are overlapping and that will be the best result for the query

    # so we will be collecting data which has a value between
    #  0.3 < data <= 1

    # cos theta = (vecQ . vecR)/(magQ . magR)

    # to convert the query and result into a higher dimensional matrix     
    # i am going to use the sentence transformer pre trained embedding model
    # i am going to use the all-MiniLM-L6-vs model of dimension 384

    def sort_service(self, query: str, search_results: list[dict]):
        relevant_docs = []
        query_embedding = self.embedding_model.encode(query)

        for result in search_results:
            if result is None:
                continue  # Skip None values
            if 'content' not in result or result['content'] is None:
                continue  # Skip if content key is missing or None

            result_embedding = self.embedding_model.encode(result["content"])

            similarity = float(np.dot(query_embedding, result_embedding)/((np.linalg.norm(query_embedding)) * (np.linalg.norm(result_embedding))))

            result["relevance_score"] = similarity

            # filtering the unrelevant docs
            if (similarity > 0.3):
                relevant_docs.append(result)

        sorted(relevant_docs, key=lambda x: x["relevance_score"], reverse=True)
        return relevant_docs