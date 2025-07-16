from config import Settings
from tavily import TavilyClient  # type: ignore
import trafilatura # type: ignore

settings = Settings()
tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)

# here i am using tavily which is used for gathering 
# web result for the given query.

# then i am using trafilatura which will help me to 
# to get more content out of the web result which we
# got from the tavily.
# i.e id we got a content multiple links or sources,
# trafilatura will download all the content from the sources.

class SearchService: 
    def web_search(self, query: str):
        results = []
        response = tavily_client.search(query, max_results=10)
        search_results = response.get("results", [])

        for result in search_results:
            downloaded = trafilatura.fetch_url(result.get("url"))
            content = trafilatura.extract(downloaded, include_comments=False)

            results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url"),
                    "content": content,
                }
            )

        return results