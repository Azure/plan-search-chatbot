import json
import logging
import os
from datetime import datetime
from typing import AsyncGenerator, List, Optional

import pytz
from config.config import Settings
from i18n.locale_msg import LOCALE_MESSAGES
from langchain.prompts import load_prompt
from model.models import ChatMessage
from openai import AsyncAzureOpenAI
from utils.enum import SearchEngine

from services.query_rewriter import QueryRewriter
from services.search_crawler import BingSearchCrawler, SearchCrawler

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

PLANNER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "planner_prompt.yaml"), encoding="utf-8"
)
ANSWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "answer_prompt.yaml"), encoding="utf-8"
)


class PlanExecutor:
    def __init__(self, settings: Settings):
        self.settings = settings
        if isinstance(settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC
        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )
        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.query_deployment_name = settings.AZURE_OPENAI_QUERY_DEPLOYMENT_NAME
        self.planner_max_plans = settings.PLANNER_MAX_PLANS
        self.query_rewriter: Optional[QueryRewriter] = None
        self.search_crawler: SearchCrawler = BingSearchCrawler()
        logger.debug(
            f"PlanExecutor initialized with Azure OpenAI deployment: {self.deployment_name}"
        )

    async def generate_response(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        query_rewrite: bool = True,
        search_engine: SearchEngine = SearchEngine.BING_SEARCH_CRAWLING,
        search_crawler: SearchCrawler = BingSearchCrawler(),
        stream: bool = False,
        elapsed_time: bool = True,
        locale: Optional[str] = "ko-KR",
    ) -> AsyncGenerator[str, None]:
        try:
            start_time = datetime.now(tz=self.timezone)
            if elapsed_time:
                logger.info(f"Starting response generation at {start_time}")
                ttft_time = None

            messages_dict = [
                {"role": msg.role, "content": msg.content} for msg in messages
            ]
            last_user_message = next(
                (
                    msg["content"]
                    for msg in reversed(messages_dict)
                    if msg["role"] == "user"
                ),
                "No question provided",
            )
            # 메시지 선택
            LOCALE_MSG = LOCALE_MESSAGES.get(locale, LOCALE_MESSAGES["ko-KR"])
            if last_user_message == "No question provided":
                yield LOCALE_MSG["input_needed"]
                return

            current_date = datetime.now(tz=self.timezone).strftime("%Y-%m-%d")

            if max_tokens is None:
                max_tokens = self.settings.MAX_TOKENS

            if temperature is None:
                temperature = self.settings.DEFAULT_TEMPERATURE

            if stream:
                yield f"data: ### {LOCALE_MSG['analyzing']}\n\n"

            enriched_query = last_user_message
            if query_rewrite and self.query_rewriter:
                rewritten_queries = await self.query_rewriter.rewrite_query(
                    last_user_message, locale=locale
                )
                enriched_query = rewritten_queries["llm_query"]

                if stream:
                    yield f"data: ### {LOCALE_MSG['analyze_complete']} \n\n"

            planner_messages = [
                {
                    "role": "system",
                    "content": PLANNER_PROMPT.format(
                        planner_max_plans=self.planner_max_plans,
                        date=current_date,
                        question=enriched_query,
                    ),
                },
                {"role": "user", "content": enriched_query},
            ]

            if stream:
                yield f"data: ### {LOCALE_MSG['plan_building']}\n\n"

            planner_response = await self.client.chat.completions.create(
                model=self.query_deployment_name,
                messages=planner_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                response_format={"type": "json_object"},
            )

            try:
                search_plan_json = json.loads(
                    planner_response.choices[0].message.content
                )
                search_queries = search_plan_json.get("search_queries", [])
            except Exception as e:
                logger.error(f"Failed to parse planner response as JSON: {e}")
                search_queries = []

            if stream:
                yield f"data: ### {LOCALE_MSG['plan_done']} {len(search_queries)}개의 검색어 생성\n\n"
                logger.debug(f"Generated search queries: {search_queries}")

            all_contexts = []
            all_url_snippet_tuples = []

            for i, query in enumerate(search_queries):
                if stream:
                    yield f"data: ### {LOCALE_MSG['searching']} ({i+1}/{len(search_queries)}): {query}\n\n"

                search_results = search_crawler.search(query, locale=locale)

                if search_results:
                    url_snippet_tuples = [
                        (r["link"], r["snippet"]) for r in search_results
                    ]
                    all_url_snippet_tuples.extend(url_snippet_tuples)
                    contexts = await search_crawler.extract_contexts_async(
                        url_snippet_tuples
                    )

                    formatted_contexts = [
                        f"[{LOCALE_MSG['search_keyword']}: {query}]\n{context}"
                        for context in contexts
                    ]

                    all_contexts.extend(formatted_contexts)

            if stream:
                yield f"data: ### {LOCALE_MSG['answering']}\n\n"

            if not all_contexts:
                yield LOCALE_MSG["no_results"]
                return

            contexts_text = "\n\n".join(all_contexts)

            answer_messages = [
                {
                    "role": "system",
                    "content": ANSWER_PROMPT.format(
                        date=current_date,
                        contexts=contexts_text,
                        question=enriched_query,
                        url_citation=json.dumps(all_url_snippet_tuples, ensure_ascii=False),
                        locale=locale,
                    ),
                },
                {"role": "user", "content": enriched_query},
            ]

            log_message = (
                f"Sending {'streaming ' if stream else ''}request to Azure OpenAI: "
                f"deployment={self.deployment_name}, "
                f"max_tokens={max_tokens}, "
                f"temperature={temperature}, "
                f"message_count={len(answer_messages)}"
            )
            logger.debug(log_message)

            response = await self.client.chat.completions.create(
                model=self.deployment_name,
                messages=answer_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stream=stream,
            )

            if stream:
                ttft_time = datetime.now(tz=self.timezone) - start_time
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield f"{chunk.choices[0].delta.content}"

            else:
                ttft_time = datetime.now(tz=self.timezone) - start_time
                message_content = response.choices[0].message.content
                yield message_content

            if elapsed_time and ttft_time is not None:
                logger.info(
                    f"Response generated successfully in {ttft_time.total_seconds()} seconds"
                )
                yield " \n\n"
                yield " \n\n"
                yield f"Response generated successfully in {ttft_time.total_seconds()} seconds \n\n"
            return
        except Exception as e:
            error_msg = (
                f"Azure OpenAI API {'streaming ' if stream else ''}error: {str(e)}"
            )
            logger.error(error_msg)
            yield f"Error: {str(e)}"
