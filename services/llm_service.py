import google.generativeai as genai # type: ignore
from config import Settings 

settings = Settings()

class LLMService:
    def __init__(self):
        genai.configure(api_key=settings.GEMINI_API_KEY)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def generate_response(self, query: str, search_results: list[dict]):
        context_text = "\n\n".join([
            f"Source {i+1} {result["url"]}:\n{result["content"]}"
            for i, result in enumerate(search_results)
        ])

        full_prompt = f"""
Context from we search:
{context_text}

Query: {query}

Please provide a comprehensive, detailed, well-cited accurate response using the above context.
Think and reason deeply. Ensure to answer the query the user is asking.
Do not use your knowledge until it is absolutely necesssary.
If the context above is empty or the context is not enough to answer the query, then use your knowledge to answer the user query.
I am ready to read your response for the above query.
"""
        
        response = self.model.generate_content(full_prompt, stream=True)

        for chunk in response:
            yield chunk.text