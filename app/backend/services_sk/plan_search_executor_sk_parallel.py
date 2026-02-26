import json
import logging
import os
from datetime import datetime
from typing import AsyncGenerator, List, Optional, Dict
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


class PlanSearchExecutorSKParallel:
    """
    Plan Search Executor using Semantic Kernel
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
            f"PlanSearchExecutor initialized with Azure OpenAI deployment: {self.deployment_name}"
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

    async def _execute_parallel_search_manual(
        self,
        search_queries: List[str],
        search_engine: SearchEngine,
        locale: str = "ko-KR",
        max_tokens: int = 2048,
        temperature: float = 0.3,
        include_web_search: bool = True,
        include_ytb_search: bool = True,
        include_mcp_server: bool = True,
        verbose: bool = False,
        LOCALE_MSG: Dict[str, str] = LOCALE_MESSAGES,
        status_callback=None,  # callback을 통해 yield 메시지 전달
    ) -> str:
        """Execute parallel search manually using asyncio.gather for better control"""
        try:
            tasks = []

            # Task 1: Web search (if enabled)
            if include_web_search and search_queries:
                tasks.append(
                    self._execute_web_search(
                        search_queries,
                        search_engine,
                        locale,
                        max_tokens,
                        temperature,
                        verbose,
                        LOCALE_MSG,
                        status_callback,
                    )
                )
            else:
                # Create a simple coroutine that returns empty string
                async def empty_web_search():
                    return ""

                tasks.append(empty_web_search())

            # Task 2: MCP search (if enabled)
            if include_ytb_search and search_queries:
                tasks.append(
                    self._execute_ytb_search(
                        search_queries,
                        include_mcp_server,
                        locale,
                        verbose,
                        LOCALE_MSG,
                        status_callback,
                    )
                )
            else:
                # Create a simple coroutine that returns empty string
                async def empty_ytb_search():
                    return ""

                tasks.append(empty_ytb_search())

            # Execute both tasks in parallel
            web_context, ytb_context = await asyncio.gather(
                *tasks, return_exceptions=True
            )

            # Handle exceptions
            if isinstance(web_context, Exception):
                logger.error(f"Web search error: {web_context}")
                web_context = f"=== Web Search ===\nError: {str(web_context)}"
            elif not web_context:
                web_context = ""

            if isinstance(ytb_context, Exception):
                logger.error(f"MCP search error: {ytb_context}")
                ytb_context = f"=== MCP ===\nError: {str(ytb_context)}"
            elif not ytb_context:
                ytb_context = ""

            # Aggregate contexts
            all_contexts = []
            if web_context and web_context.strip():
                all_contexts.append(web_context)
            if ytb_context and ytb_context.strip():
                all_contexts.append(ytb_context)

            if not all_contexts:
                all_contexts.append("No relevant context found.")

            return "\n".join(all_contexts)

        except Exception as e:
            logger.error(f"Error in parallel search execution: {e}")
            return f"Error in parallel search: {str(e)}"

    async def _execute_web_search(
        self,
        search_queries: List[str],
        search_engine: SearchEngine,
        locale: str,
        max_tokens: int,
        temperature: float,
        verbose: bool = False,
        LOCALE_MSG: Dict[str, str] = LOCALE_MESSAGES,
        status_callback=None,
    ) -> str:
        """Execute web search with full parallelization"""
        try:
            web_search_contexts = []

            if search_engine == SearchEngine.BING_GROUNDING:

                for i, query in enumerate(search_queries):
                    if status_callback:
                        status_callback(
                            f"data: ### {LOCALE_MSG['searching']} ({i+1}/{len(search_queries)}): {query}\n\n"
                        )

                # Use grounding plugin (single call for multiple queries)
                grounding_function = self.kernel.get_function(
                    "grounding", "grounding_search_multi_query"
                )
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
                    web_search_contexts.append(grounding_result.value)

            elif search_engine in [
                SearchEngine.BING_SEARCH_CRAWLING,
                SearchEngine.BING_GROUNDING_CRAWLING,
            ]:

                search_function = self.kernel.get_function(
                    "search", "search_single_query"
                )

                # invoke Web search function for each query at once
                search_tasks = []
                for i, query in enumerate(search_queries):
                    if status_callback:
                        status_callback(
                            f"data: ### {LOCALE_MSG['searching']} ({i+1}/{len(search_queries)}): {query}\n\n"
                        )

                    task = search_function.invoke(
                        self.kernel,
                        KernelArguments(
                            query=query,
                            locale=locale,
                            max_results=5,
                            max_context_length=5000,
                        ),
                    )
                    search_tasks.append(task)

                # Parallel process using asyncio.gather
                search_results = await asyncio.gather(
                    *search_tasks, return_exceptions=True
                )

                # get the contexts from search results
                for i, result in enumerate(search_results):
                    if isinstance(result, Exception):
                        logger.error(f"Search error for query {i}: {result}")
                    elif result and result.value:
                        web_search_contexts.append(result.value)

            if web_search_contexts:
                combined_web_context = "\n\n".join(web_search_contexts)
                if verbose and status_callback:
                    status_callback(
                        f"data: {self.send_step_with_code(LOCALE_MSG['search_done'], combined_web_context)}\n\n"
                    )
                return f"=== Web Search ===\n{combined_web_context}"
            else:
                if status_callback:
                    status_callback(
                        "data: ### Web search completed with no results\n\n"
                    )
                return ""

        except Exception as e:
            logger.error(f"Error in web search: {e}")
            if status_callback:
                status_callback(f"data: ### Web search error: {str(e)}\n\n")
            raise e

    async def _execute_ytb_search(
        self,
        search_queries: List[str],
        include_mcp_server: bool,
        locale: str,
        verbose: bool = False,
        LOCALE_MSG: Dict[str, str] = LOCALE_MESSAGES,
        status_callback=None,
    ) -> str:
        """Execute YouTube search with full parallelization"""
        try:
            youtube_search_contexts = []

            # invoke YouTube search function for each query at once
            youtube_tasks = []
            for i, query in enumerate(search_queries):
                if status_callback:
                    status_callback(
                        f"data: ### {LOCALE_MSG['searching_YouTube']} ({i+1}/{len(search_queries)}): {query}\n\n"
                    )

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

                task = youtube_search_function.invoke(self.kernel, mcp_args)
                youtube_tasks.append(task)

            # Parallel process using asyncio.gather
            youtube_results = await asyncio.gather(
                *youtube_tasks, return_exceptions=True
            )

            # get the contexts from search results
            for i, result in enumerate(youtube_results):
                if isinstance(result, Exception):
                    logger.error(f"YouTube search error for query {i}: {result}")
                elif result and result.value:
                    youtube_search_contexts.append(result.value)
                    logger.info(
                        f"Added youtube search result for query: {search_queries[i]}"
                    )
                else:
                    logger.warning(
                        f"No youtube search result for query: {search_queries[i]}"
                    )

            if youtube_search_contexts:
                combined_youtube_context = "\n\n".join(youtube_search_contexts)
                if verbose and status_callback:
                    status_callback(
                        f"data: {self.send_step_with_code(LOCALE_MSG['YouTube_done'], combined_youtube_context)}\n\n"
                    )
                return f"=== Youtube Search ===\n{combined_youtube_context}"
            else:
                return ""

        except Exception as e:
            logger.error(f"Error in YouTube search: {e}")
            if status_callback:
                status_callback(f"data: ### YouTube search error: {str(e)}\n\n")
            raise e

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
            planning: Whether to include planning
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
                logger.info(f"Starting plan search response generation at {start_time}")
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

            # Get localWe messages
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
            user_intent = "general_query"  # Initialize user_intent
            intent_data = {}
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
                            current_date=current_date,  # 추가: current_date 파라미터
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

                        logger.info("=" * 60)
                        logger.info("Intent analysis result:")
                        logger.info(f"User intent: {user_intent}")
                        logger.info(f"Enriched query: {enriched_query}")
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
                        include_mcp_server = False

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
                                current_date=current_date,  # 추가: current_date 파라미터 (필요시)
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

            logger.info("Starting parallel search execution...")

            # Create a queue to collect status messages from parallel tasks
            status_queue = asyncio.Queue()

            # Define callback function that puts messages in queue
            def status_callback(message):
                try:
                    status_queue.put_nowait(message)
                except asyncio.QueueFull:
                    logger.warning(
                        f"Status queue full, dropping message: {message[:50]}..."
                    )

            # Start the parallel search task
            search_task = asyncio.create_task(
                self._execute_parallel_search_manual(
                    search_queries=search_queries,
                    search_engine=search_engine,
                    locale=locale,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    verbose=verbose,
                    LOCALE_MSG=LOCALE_MSG,
                    include_web_search=include_web_search,
                    include_ytb_search=include_ytb_search,  # Use include_ytb_search instead of include_mcp_server
                    include_mcp_server=include_mcp_server,
                    status_callback=status_callback,
                )
            )

            # Process status messages while search is running
            all_contexts = []
            while not search_task.done():
                try:
                    # Check for status messages with a short timeout
                    message = await asyncio.wait_for(status_queue.get(), timeout=0.1)
                    if stream:
                        yield message
                except asyncio.TimeoutError:
                    # No message available, continue waiting
                    continue
                except Exception as e:
                    logger.error(f"Error processing status message: {e}")
                    break

            # Get the final result
            try:
                context = await search_task
                all_contexts.append(context)
            except Exception as e:
                logger.error(f"Error in search task: {e}")
                all_contexts = [f"Error in parallel search: {str(e)}"]

            # Process any remaining status messages
            while not status_queue.empty():
                try:
                    message = status_queue.get_nowait()
                    if stream:
                        yield message
                except asyncio.QueueEmpty:
                    break

            logger.info("Parallel search execution completed")

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
            # YouTubeMCPPlugin 정리
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
