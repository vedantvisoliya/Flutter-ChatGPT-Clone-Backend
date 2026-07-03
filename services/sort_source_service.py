import google.generativeai as genai  # type: ignore
import numpy as np  # type: ignore
from config import Settings

settings = Settings()
genai.configure(api_key=settings.GEMINI_API_KEY)

# Gemini's embedding model. Using the hosted embedding API keeps the
# service lightweight (no torch / sentence-transformers), so it fits well
# within Render's free 512MB instance.
EMBEDDING_MODEL = "models/text-embedding-004"


class SortSourceService:

    # i am using COS SIMILARITY SEARCH

    # what is embedding or vectorisation?
    # -> in this case we have query which is converted into
    # a high order or high dimension space matrix

    # in cos similarity search we check the angle between
    # query vector and result vector and check there similarity
    # that the content we got from the web and check is the content
    # a valid result for the query.

    # in cos theta case lesser the theta greater the value
    # that means if theta is 0 degree i.e value will be 1 which implies the
    # vectors are overlapping and that will be the best result for the query

    # so we will be collecting data which has a value between
    #  0.3 < data <= 1

    # cos theta = (vecQ . vecR)/(magQ . magR)

    def _embed(self, text: str, task_type: str) -> np.ndarray:
        response = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text,
            task_type=task_type,
        )
        return np.array(response["embedding"])

    def sort_service(self, query: str, search_results: list[dict]):
        relevant_docs = []
        query_embedding = self._embed(query, task_type="retrieval_query")

        for result in search_results:
            if result is None:
                continue  # Skip None values
            if 'content' not in result or result['content'] is None:
                continue  # Skip if content key is missing or None

            result_embedding = self._embed(result["content"], task_type="retrieval_document")

            similarity = float(
                np.dot(query_embedding, result_embedding)
                / ((np.linalg.norm(query_embedding)) * (np.linalg.norm(result_embedding)))
            )

            result["relevance_score"] = similarity

            # filtering the unrelevant docs
            if (similarity > 0.3):
                relevant_docs.append(result)

        relevant_docs.sort(key=lambda x: x["relevance_score"], reverse=True)
        return relevant_docs
