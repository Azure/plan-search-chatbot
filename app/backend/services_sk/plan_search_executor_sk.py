import json
import logging
import os
from datetime import datetime
from typing import AsyncGenerator, List, Optional
import asyncio
import pytz
from config.config import Settings
from i18n.locale_msg import LOCALE_MESSAGES
from langchain.prompts import load_prompt
from model.models import ChatMessage
from openai import AsyncAzureOpenAI
from utils.enum import SearchEngine
import base64

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_arguments import KernelArguments

from .search_plugin import SearchPlugin
from services_sk.youtube_plugin import YouTubePlugin
from services_sk.youtube_mcp_plugin import YouTubeMCPPlugin
from .corp_plugin import CORPPlugin
from .intent_plan_plugin import IntentPlanPlugin
from .grounding_plugin import GroundingPlugin

logger = logging.getLogger(__name__)

current_dir = os.path.dirname(os.path.abspath(__file__))

# Load prompts
SEARCH_PLANNER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "planner_prompt.yaml"), encoding="utf-8"
)
PRODUCT_ANSWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "product_answer_prompt.yaml"),
    encoding="utf-8",
)
GENERAL_ANSWER_PROMPT = load_prompt(
    os.path.join(current_dir, "..", "prompts", "general_answer_prompt.yaml"),
    encoding="utf-8",
)


class PlanSearchExecutorSK:
    """
    Plan and Search Executor using Semantic Kernel.
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        if isinstance(settings.TIME_ZONE, str):
            self.timezone = pytz.timezone(settings.TIME_ZONE)
        else:
            self.timezone = pytz.UTC

        # Initialize OpenAI client for legacy operations
        self.client = AsyncAzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )

        # Initialize Semantic Kernel
        self.kernel = Kernel()

        # Add Azure OpenAI chat completion service
        self.chat_completion = AzureChatCompletion(
            deployment_name=settings.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_key=settings.AZURE_OPENAI_API_KEY,
            base_url=settings.AZURE_OPENAI_ENDPOINT,
            api_version=settings.AZURE_OPENAI_API_VERSION,
        )
        self.kernel.add_service(self.chat_completion)

        # Initialize plugins
        bing_api_key = getattr(settings, "BING_API_KEY", None)
        bing_endpoint = getattr(settings, "BING_ENDPOINT", None)

        logger.info("Initializing SearchPlugin with:")
        logger.info(
            f"  - bing_api_key from settings: {'SET' if bing_api_key else 'NOT SET'}"
        )
        logger.info(f"  - bing_endpoint from settings: {bing_endpoint}")

        self.search_plugin = SearchPlugin(
            bing_api_key=bing_api_key, bing_endpoint=bing_endpoint
        )
        self.youtube_plugin = YouTubePlugin()
        self.youtube_mcp_plugin = YouTubeMCPPlugin()
        self.corp_plugin = CORPPlugin()
        self.intent_plan_plugin = IntentPlanPlugin(settings)
        self.grounding_plugin = GroundingPlugin()

        # Add plugins to kernel
        self.kernel.add_plugin(self.search_plugin, plugin_name="search")
        self.kernel.add_plugin(self.grounding_plugin, plugin_name="grounding")
        self.kernel.add_plugin(self.youtube_plugin, plugin_name="youtube")
        self.kernel.add_plugin(self.youtube_mcp_plugin, plugin_name="youtube_mcp")
        self.kernel.add_plugin(self.corp_plugin, plugin_name="corp")  # not use anymore
        self.kernel.add_plugin(self.intent_plan_plugin, plugin_name="intent_plan")

        self.deployment_name = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.query_deployment_name = settings.AZURE_OPENAI_QUERY_DEPLOYMENT_NAME
        self.planner_max_plans = settings.PLANNER_MAX_PLANS

        logger.debug(
            f"RiskSearchExecutor initialized with Azure OpenAI deployment: {self.deployment_name}"
        )

    @staticmethod
    def send_step_with_code(step_name: str, code: str) -> str:
        """Send a step with code content"""
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        return f"### {step_name}#code#{encoded_code}"

    @staticmethod
    def send_step_with_input(step_name: str, description: str) -> str:
        """Send a step with input description"""
        return f"### {step_name}#input#{description}"

    @staticmethod
    def send_step_with_code_and_input(
        step_name: str, code: str, description: str
    ) -> str:
        """Send a step with both code and input description"""
        encoded_code = base64.b64encode(code.encode("utf-8")).decode("utf-8")
        return f"### {step_name}#input#{description}#code#{encoded_code}"

    async def generate_response(
        self,
        messages: List[ChatMessage],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        query_rewrite: bool = True,
        planning: bool = True,
        search_engine: SearchEngine = SearchEngine.BING_SEARCH_CRAWLING,
        stream: bool = False,
        elapsed_time: bool = True,
        locale: Optional[str] = "ko-KR",
        include_web_search: bool = True,
        include_ytb_search: bool = True,
        include_mcp_server: bool = True,
        verbose: Optional[bool] = False,
    ) -> AsyncGenerator[str, None]:
        """
        Generate response using semantic kernel with search and/or MCP plugins.

        Args:
            messages: Chat messages history
            max_tokens: Maximum tokens for response
            temperature: Temperature for response generation
            query_rewrite: Whether to rewrite the query
            planning: Whether to use planning to search
            search_engine: Search engine to use
            stream: Whether to stream response
            elapsed_time: Whether to include elapsed time
            locale: Locale for search and response
            include_web_search: Whether to include web search results
            include_ytb_search: Whether to include YouTube search results
            include_mcp_server: Whether to include MCP server integration
            verbose: Whether to include verbose context information,

        """
        try:
            start_time = datetime.now(tz=self.timezone)
            if elapsed_time:
                logger.info(f"Starting risk search response generation at {start_time}")
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

            # Get locale messages
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

            # Intent analysis and query rewriting using IntentPlugin
            enriched_query = last_user_message
            search_queries = []
            resource_group_name = None
            user_intent = "general_query"
            intent_data = None
            plan_data = None

            if query_rewrite:
                try:
                    # Use IntentPlugin for intent analysis
                    intent_function = self.kernel.get_function(
                        "intent_plan", "analyze_intent"
                    )
                    intent_result = await intent_function.invoke(
                        self.kernel,
                        KernelArguments(
                            original_query=last_user_message,
                            locale=locale,
                            temperature=0.3,
                        ),
                    )

                    if intent_result and intent_result.value:
                        intent_data = json.loads(intent_result.value)
                        user_intent = intent_data.get("user_intent", "general_query")
                        enriched_query = intent_data.get(
                            "enriched_query", last_user_message
                        )
                        resource_group_name = intent_data.get("resource_group_name")

                        logger.info("=" * 60)
                        logger.info("Intent analysis result:")
                        logger.info(f"User intent: {user_intent}")
                        logger.info(f"Enriched query: {enriched_query}")
                        logger.info(f"resource group name: {resource_group_name}")
                        logger.info("=" * 60)

                    if verbose and stream:
                        intent_data_str = (
                            json.dumps(intent_data, ensure_ascii=False, indent=2)
                            if intent_data
                            else "{}"
                        )
                        yield f"data: {self.send_step_with_code(LOCALE_MSG['analyze_complete'], intent_data_str)}\n\n"

                    if user_intent == "small_talk":
                        # Small talk does not require search
                        planning = False
                        include_web_search = False
                        include_ytb_search = False

                        if stream:
                            yield f"data: ### {LOCALE_MSG['intent_small_talk']}\n\n"

                    if planning:

                        if stream:
                            yield f"data: ### {LOCALE_MSG['search_planning']}\n\n"

                        # Generate search plan using IntentPlanPlugin
                        plan_function = self.kernel.get_function(
                            "intent_plan", "generate_search_plan"
                        )
                        plan_result = await plan_function.invoke(
                            self.kernel,
                            KernelArguments(
                                user_intent=user_intent,
                                enriched_query=enriched_query,
                                locale=locale,
                                temperature=0.7,
                            ),
                        )

                        if plan_result and plan_result.value:
                            plan_data = json.loads(plan_result.value)
                            search_queries = plan_data.get(
                                "search_queries", [enriched_query]
                            )

                            logger.info(f"Search plan: {plan_data}")
                        else:
                            # Fallback
                            search_queries = [enriched_query]

                        if verbose and stream:
                            plan_data_str = (
                                json.dumps(plan_data, ensure_ascii=False, indent=2)
                                if plan_data
                                else "{}"
                            )
                            yield f"data: {self.send_step_with_code(LOCALE_MSG['plan_done'], plan_data_str)}\n\n"

                    else:
                        # Fallback
                        search_queries = [enriched_query]

                except Exception as e:
                    logger.error(f"Error during intent analysis: {e}")
                    # Fallback to original query
                    search_queries = [enriched_query]
                    if stream:
                        yield "data: ### Intent analysis failed, using fallback\n\n"
            else:
                # No query rewriting
                search_queries = [enriched_query]

            # Collect contexts
            all_contexts = []

            # Web search context
            if include_web_search and search_queries:

                try:
                    if search_engine == SearchEngine.BING_GROUNDING:
                        # Use grounding plugin for BING_GROUNDING
                        logger.info("Using GroundingPlugin for BING_GROUNDING search")

                        text_appended_query = f"{LOCALE_MSG['searching']}... "
                        for i, query in enumerate(search_queries):
                            text_appended_query += (
                                f"{i}: {LOCALE_MSG['search_keyword']}: {query} "
                            )

                        if stream:
                            yield f"data: ### {text_appended_query}\n\n"

                        grounding_function = self.kernel.get_function(
                            "grounding", "grounding_search_multi_query"
                        )

                        # Convert search_queries list to JSON string for the plugin
                        search_queries_json = json.dumps(search_queries)

                        grounding_result = await grounding_function.invoke(
                            self.kernel,
                            KernelArguments(
                                search_queries=search_queries_json,
                                max_tokens=max_tokens,
                                temperature=temperature,
                                locale=locale,
                            ),
                        )

                        if grounding_result and grounding_result.value:
                            all_contexts.append(
                                f"=== Grounding Search ===\n{grounding_result.value}"
                            )
                            logger.info("Successfully got grounding search results")
                        else:
                            logger.warning("No grounding search results obtained")

                    elif (
                        search_engine == SearchEngine.BING_SEARCH_CRAWLING
                        or search_engine == SearchEngine.BING_GROUNDING_CRAWLING
                    ):
                        # Use existing search plugin for other search engines
                        logger.info(f"Using SearchPlugin for {search_engine} search")

                        web_search_contexts = []
                        search_function = self.kernel.get_function(
                            "search", "search_single_query"
                        )

                        # Process each search query individually
                        for i, query in enumerate(search_queries):
                            if stream:
                                yield f"data: ### {LOCALE_MSG['searching']} ({i+1}/{len(search_queries)}): {query}\n\n"

                            logger.info(f"Invoking search function for query: {query}")

                            search_result = await search_function.invoke(
                                self.kernel,
                                KernelArguments(
                                    query=query,
                                    locale=locale,
                                    max_results=5,  # Limit to 5 results per query
                                    max_context_length=5000,  # Limit context length for scrapping
                                ),
                            )

                            if search_result:
                                logger.info(
                                    f"Search result value length: {len(str(search_result.value)) if search_result.value else 0}"
                                )
                                logger.info(
                                    f"Search result value preview: {str(search_result.value)[:200] if search_result.value else 'None'}"
                                )

                            if search_result and search_result.value:
                                web_search_contexts.append(search_result.value)
                                logger.info(
                                    f"Added web search result for query: {query}"
                                )
                            else:
                                logger.warning(
                                    f"No web search result for query: {query}"
                                )

                        if web_search_contexts:
                            combined_web_context = "\n\n".join(web_search_contexts)
                            all_contexts.append(
                                f"=== Web Search ===\n{combined_web_context}"
                            )

                    if verbose and stream and web_search_contexts:
                        yield f"data: {self.send_step_with_code(LOCALE_MSG['search_done'], combined_web_context)}\n\n"

                except Exception as e:
                    logger.error(f"Error during web search: {e}")
                    if stream:
                        yield f"data: ### Error during web search: {str(e)}\n\n"

            # MCP server context
            if include_ytb_search:
                try:
                    youtube_search_contexts = []
                    # Process each search query individually
                    for i, query in enumerate(search_queries):
                        if stream:
                            yield f"data: ### {LOCALE_MSG['searching_YouTube']} ({i+1}/{len(search_queries)}): {query}\n\n"

                        # Use kernel to invoke youtube / youtube_mcp plugin
                        if include_mcp_server:
                            youtube_search_function = self.kernel.get_function(
                                "youtube_mcp", "search_youtube_videos"
                            )
                        else:
                            youtube_search_function = self.kernel.get_function(
                                "youtube", "search_youtube_videos"
                            )
                        mcp_args = KernelArguments()
                        mcp_args["query"] = query

                        mcp_result = await youtube_search_function.invoke(
                            self.kernel, mcp_args
                        )

                        if mcp_result and mcp_result.value:

                            if mcp_result and mcp_result.value:
                                youtube_search_contexts.append(mcp_result.value)
                                logger.info(
                                    f"Added youtube search result for query: {query}"
                                )
                            else:
                                logger.warning(
                                    f"No youtube search result for query: {query}"
                                )

                            # Check if the result contains an error message
                        else:
                            logger.warning("No Youtube Search result returned")

                    if youtube_search_contexts:
                        combined_youtube_context = "\n\n".join(
                            youtube_search_contexts
                        )
                        all_contexts.append(
                            f"=== Youtube Search ===\n{combined_youtube_context}"
                        )

                    if verbose and stream and youtube_search_contexts:
                        yield f"data: {self.send_step_with_code(LOCALE_MSG['YouTube_done'], combined_youtube_context)}\n\n"

                except Exception as e:
                    logger.error(f"Error during Youtube MCP search: {e}")
                    if stream:
                        yield f"data: ### : Error during Youtube MCP search: {str(e)}\n\n"

            if stream:
                yield f"data: ### {LOCALE_MSG['answering']}\n\n"

            if not all_contexts:
                all_contexts.append("No relevant context found.")

            contexts_text = "\n".join(all_contexts)

            yield " \n"  # clear previous md formatting

            # Generate final answer
            if user_intent == "general_query":
                answer_messages = [
                    {
                        "role": "system",
                        "content": GENERAL_ANSWER_PROMPT.format(
                            current_date=current_date,
                            contexts=contexts_text,
                            question=enriched_query,
                            locale=locale,
                        ),
                    },
                    {"role": "user", "content": enriched_query},
                ]
            elif user_intent == "product_query":
                answer_messages = [
                    {
                        "role": "system",
                        "content": PRODUCT_ANSWER_PROMPT.format(
                            current_date=current_date,
                            contexts=contexts_text,
                            question=enriched_query,
                            locale=locale,
                        ),
                    },
                    {"role": "user", "content": enriched_query},
                ]
            else:
                answer_messages = [
                    {
                        "role": "system",
                        "content": GENERAL_ANSWER_PROMPT.format(
                            current_date=current_date,
                            contexts=contexts_text,
                            question=enriched_query,
                            locale=locale,
                        ),
                    },
                    {"role": "user", "content": enriched_query},
                ]

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

            yield "\n"  # clear previous md formatting

            if elapsed_time and ttft_time is not None:
                logger.info(
                    f"Plan search response generated successfully in {ttft_time.total_seconds()} seconds"
                )
                yield "\n"
                yield f"Plan search response generated successfully in {ttft_time.total_seconds()} seconds \n"

        except Exception as e:
            error_msg = f"Plan search error: {str(e)}"
            logger.error(error_msg)
            yield f"Error: {str(e)}"

    async def cleanup(self):
        """Clean up resources"""
        try:
            # AzureMCPPlugin 정리
            if hasattr(self.youtube_plugin, "cleanup"):
                await self.youtube_plugin.cleanup()
            if hasattr(self.youtube_mcp_plugin, "cleanup"):
                await self.youtube_mcp_plugin.cleanup()

            # IntentPlugin 정리
            if hasattr(self.intent_plan_plugin, "cleanup"):
                await self.intent_plan_plugin.cleanup()

            # GroundingPlugin 정리
            if hasattr(self.grounding_plugin, "cleanup"):
                await self.grounding_plugin.cleanup()

            # OpenAI 클라이언트 정리
            if hasattr(self.client, "close"):
                await self.client.close()

            # 잠시 대기하여 연결이 완전히 정리되도록 함
            await asyncio.sleep(0.1)

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
