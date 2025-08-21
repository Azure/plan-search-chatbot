import chainlit as cl
import requests
import os
import sys
import json
import logging
import base64
from datetime import datetime
from typing import Dict, List, Optional, Any
from i18n.locale_msg_front import UI_TEXT, EXAMPLE_PROMPTS
from pathlib import Path

# 간단한 사용자 인증 설정
AUTH_USERNAME = os.getenv("AUTH_USERNAME", "ms_user")
AUTH_PASSWORD = os.getenv("AUTH_PASSWORD", "msuser123")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# @cl.password_auth_callback
def auth_callback(username: str, password: str):
    """Simple password authentication - fixed version"""
    try:
        logger.info(f"🔐 Authentication attempt - username: {username}")
        
        # MS 사용자 인증
        if username == AUTH_USERNAME and password == AUTH_PASSWORD:
            logger.info("✅ MS user authentication successful")
            return cl.User(
                identifier="ms_user",
                metadata={
                    "role": "user",
                    "name": "Microsoft User",
                    "login_time": datetime.now().isoformat()
                }
            )
        
        # 관리자 인증
        elif username == "admin" and password == ADMIN_PASSWORD:
            logger.info("✅ Admin authentication successful")
            return cl.User(
                identifier="admin",
                metadata={
                    "role": "admin", 
                    "name": "Administrator",
                    "login_time": datetime.now().isoformat()
                }
            )
        
        logger.warning(f"❌ Authentication failed for user: {username}")
        return None
        
    except Exception as e:
        logger.error(f"❌ Authentication error: {e}")
        return None

# Load environment variables
SK_API_URL = os.getenv("SK_API_URL", "http://localhost:8000/plan_search")
SK_API_URL_PARALLEL = os.getenv("SK_API_URL_PARALLEL", "http://localhost:8000/plan_search_parallel")

# Define the search engines
SEARCH_ENGINES = {
    "Bing Search": "bing_search_crawling",
    "Grounding Gen": "grounding_bing"
}

# Internationalization constants
SUPPORTED_LANGUAGES = {
    "en-US": "English",
    "ko-KR": "한국어"
}

class ChatSettings:
    """Chat settings for managing user preferences"""
    def __init__(self):
        self.query_rewrite = True
        self.web_search = True
        self.planning = True
        self.ytb_search = True
        self.mcp_server = True
        self.verbose = True
        self.parallel = True
        self.search_engine = list(SEARCH_ENGINES.values())[0]
        self.language = "en-US"
        self.max_tokens = 4000
        self.temperature = 0.7

def get_current_prompt(lang: str, category: str) -> str:
    """Get current prompt text for a category in the specified language"""
    return EXAMPLE_PROMPTS[lang][category]["prompt"]

def get_starter_label(lang: str, category: str) -> str:
    """Get starter label for a category in the specified language"""
    return EXAMPLE_PROMPTS[lang][category]["title"]

def get_starters_for_language(language: str):
    """Get starters for a specific language"""
    starters = []
    
    categories = ["question_Microsoft", "product_info", "recommendation", "comparison", "support_questions", "tools"]
    logger.info(f"Getting starters for language: {language}")
    logger.info(f"Available categories in EXAMPLE_PROMPTS: {list(EXAMPLE_PROMPTS.get(language, {}).keys())}")
    
    for category in categories:
        if category in EXAMPLE_PROMPTS[language]:
            if category == "question_Microsoft":
                emoji="📈" 
                image="/public/images/1f4c8_color.png"
            elif category == "product_info":
                emoji="✅"
                image="/public/images/2705_flat.png"
            elif category == "recommendation":
                emoji="💡"
                image="/public/images/1f4a1_color.png"
            elif category == "comparison":
                emoji="📚"
                image="/public/images/1f4da_color.png"
            elif category == "support_questions":
                emoji="👨‍💻"
                image="/public/images/1f468-1f4bb_flat.png"
            elif category == "tools":
                emoji="🛠"
                image="/public/images/1f6e0_color.png"
                        
            starter = cl.Starter(
                label=get_starter_label(language, category),
                message=get_current_prompt(language, category),
                icon=image
            )
            starters.append(starter)
            logger.info(f"Added starter: {category} - {starter.label}")
        else:
            logger.warning(f"Category {category} not found in EXAMPLE_PROMPTS[{language}]")
    
    logger.info(f"Total starters created: {len(starters)}")
    return starters

@cl.set_chat_profiles
async def chat_profile():
    """Set up chat profiles for different languages"""
    return [
        cl.ChatProfile(
            name="English", 
            markdown_description="## Plan Search Chat",
            icon="/public/images/ai_foundry_icon_small.png",
            starters=get_starters_for_language("en-US")
        ),
        cl.ChatProfile(
            name="Korean",
            markdown_description="## Plan Search Chat",
            icon="/public/images/ai_foundry_icon_small.png",
            starters=get_starters_for_language("ko-KR")
        ),
    ]

@cl.on_chat_start
async def start():
    """Initialize chat session with user welcome"""
    # 사용자 정보 가져오기
    user = cl.user_session.get("user")
    
    # 사용자 환영 메시지
    if user:
        user_role = user.metadata.get("role", "user")
        
        # 관리자 권한이 있는 경우 추가 메시지
        if user_role == "admin":
            await cl.Message(content="🔧 **Admin Access Granted**\nYou have administrator privileges.").send()
    
    # Get current chat profile
    profile = cl.user_session.get("chat_profile", "English")
    language = "ko-KR" if profile == "Korean" else "en-US"
    
    # Initialize chat settings
    settings = ChatSettings()
    settings.language = language
    cl.user_session.set("settings", settings)
    
    # Set up chat settings UI
    ui_text = UI_TEXT[language]
    
    # Create settings components
    settings_components = [
        cl.input_widget.Switch(
            id="query_rewrite",
            label=ui_text["query_rewrite_title"],
            initial=True,
            tooltip=ui_text["query_rewrite_desc"]
        ),
        cl.input_widget.Switch(
            id="web_search",
            label=ui_text["web_search_title"],
            initial=True,
            tooltip=ui_text["web_search_desc"]
        ),
        cl.input_widget.Switch(
            id="planning",
            label=ui_text["planning_title"],
            initial=True,
            tooltip=ui_text["planning_desc"]
        ),
        cl.input_widget.Switch(
            id="ytb_search",
            label=ui_text["ytb_search_title"],
            initial=True,
            tooltip=ui_text["ytb_search_desc"]
        ),
        cl.input_widget.Switch(
            id="mcp",
            label=ui_text["mcp_title"],
            initial=True,
            tooltip=ui_text["mcp_desc"]
        ),
        cl.input_widget.Switch(
            id="verbose",
            label=ui_text["verbose_title"],
            initial=True,
            tooltip=ui_text["verbose_desc"]
        ),
        cl.input_widget.Switch(
            id="parallel",
            label=ui_text["parallel_title"],
            initial=True,
            tooltip=ui_text["parallel_desc"]
        ),
        cl.input_widget.Select(
            id="search_engine",
            label=ui_text["search_engine_title"],
            values=list(SEARCH_ENGINES.keys()),
            initial_index=0,
            tooltip=ui_text["search_engine_desc"]
        ),
        cl.input_widget.Switch(
            id="show_starters",
            label="📋 Show Quick Start Options",
            initial=False,
            tooltip="Toggle to show/hide quick start prompts"
        ),
        cl.input_widget.Slider(
            id="max_tokens",
            label="Max Tokens",
            initial=4000,
            min=1000,
            max=8000,
            step=500,
            tooltip="Maximum number of tokens in response"
        ),
        cl.input_widget.Slider(
            id="temperature",
            label="Temperature",
            initial=0.7,
            min=0.0,
            max=1.0,
            step=0.1,
            tooltip="Controls randomness in response generation"
        )
    ]
    
    # Send settings to user
    await cl.ChatSettings(settings_components).send()
    
    # Set first message flag
    cl.user_session.set("first_message", True)

@cl.on_settings_update
async def setup_agent(settings_dict: Dict[str, Any]):
    """Update settings when user changes them"""
    settings = cl.user_session.get("settings")
    
    # Update settings based on user input
    settings.query_rewrite = settings_dict.get("query_rewrite", True)
    settings.planning = settings_dict.get("planning", True)
    settings.web_search = settings_dict.get("web_search", True)
    settings.ytb_search = settings_dict.get("ytb_search", True)
    settings.mcp_server = settings_dict.get("mcp_server", True)
    settings.verbose = settings_dict.get("verbose", True)
    settings.parallel = settings_dict.get("parallel", True)
    settings.max_tokens = settings_dict.get("max_tokens", 4000)
    settings.temperature = settings_dict.get("temperature", 0.7)
    
    # Update search engine
    search_engine_name = settings_dict.get("search_engine", list(SEARCH_ENGINES.keys())[0])
    settings.search_engine = SEARCH_ENGINES.get(search_engine_name, list(SEARCH_ENGINES.values())[0])
    
    # Check if user wants to show starters
    show_starters = settings_dict.get("show_starters", False)
    if show_starters:
        # Re-send starters
        current_profile = cl.user_session.get("chat_profile", "English")
        language = "ko-KR" if current_profile == "Korean" else "en-US"
        starters = get_starters_for_language(language)
        
        # Send starters as a message with action buttons
        starters_message = "📋 **Quick Start Options:**\n\n"
        actions = []
        
        for i, starter in enumerate(starters):
            actions.append(
                cl.Action(
                    name=f"starter_{i}",
                    payload={"message": starter.message, "label": starter.label},
                    label=starter.label,
                    description=f"Use starter: {starter.label}"
                )
            )
        
        await cl.Message(content=starters_message, actions=actions).send()
    
    cl.user_session.set("settings", settings)
    
    # Send confirmation message
    ui_text = UI_TEXT[settings.language]
    await cl.Message(content="⚙️ Settings updated successfully!").send()

async def safe_stream_token(msg: cl.Message, content: str) -> bool:
    """Safely stream token with connection check"""
    try:
        await msg.stream_token(content)
        return True
    except Exception as e:
        logger.warning(f"Failed to stream token: {str(e)}")
        return False

async def safe_send_step(step: cl.Step) -> bool:
    """Safely send step with connection check"""
    try:
        await step.send()
        return True
    except Exception as e:
        logger.warning(f"Failed to send step: {str(e)}")
        return False

async def safe_update_message(msg: cl.Message) -> bool:
    """Safely update message with connection check"""
    try:
        await msg.update()
        return True
    except Exception as e:
        logger.warning(f"Failed to update message: {str(e)}")
        return False

def decode_step_content(content: str) -> tuple[str, str, str]:
    """
    Decode step content that may contain code or input data
    Returns: (step_name, code_content, description)
    """
    step_name = content
    code_content = ""
    description = ""
    
    logger.info(f"Decoding step content: {content}")
    
    # Check for code content (Base64 encoded)
    if '#code#' in content:
        parts = content.split('#code#')
        step_name = parts[0]
        if len(parts) > 1:
            try:
                encoded_code = parts[1]
                logger.info(f"Found encoded code: {encoded_code[:50]}...")
                code_content = base64.b64decode(encoded_code).decode('utf-8')
                logger.info(f"Decoded code: {code_content[:100]}...")
            except Exception as e:
                logger.warning(f"Failed to decode code content: {e}")
                code_content = parts[1]  # fallback to raw content
    
    # Check for input description
    if '#input#' in step_name:
        parts = step_name.split('#input#')
        step_name = parts[0]
        if len(parts) > 1:
            description = parts[1].strip()
    
    logger.info(f"Decoded result - step_name: {step_name}, code_length: {len(code_content)}, description: {description}")
    
    return step_name, code_content, description

async def stream_chat_with_api(message: str, settings: ChatSettings) -> None:
    """Stream-enabled chat function that yields partial updates using Chainlit's Step API"""
    if not message or message.strip() == "":
        return
    
    # Get conversation history
    message_history = cl.chat_context.to_openai()
    
    # Prepare the API payload
    payload = {
        "messages": message_history[-10:],
        "max_tokens": settings.max_tokens,
        "temperature": settings.temperature,
        "query_rewrite": settings.query_rewrite,
        "planning": settings.planning,
        "include_web_search": settings.web_search,
        "include_ytb_search": settings.ytb_search,
        "include_mcp_server": settings.mcp_server,
        "search_engine": settings.search_engine,
        "stream": True,
        "locale": settings.language,
        "verbose": settings.verbose,
    }
    
    # Debug logging
    logger.info(f"API Payload: query_rewrite={settings.query_rewrite}, web_search={settings.web_search}, planning={settings.planning},"
          f"ytb_search={settings.ytb_search}, mcp_server={settings.mcp_server}, search_engine={settings.search_engine}, "
          f"max_tokens={settings.max_tokens}, temperature={settings.temperature}, "
          f"language={settings.language}, verbose={settings.verbose}")
    
    # Create message for streaming response
    ui_text = UI_TEXT[settings.language]
    msg = cl.Message(content="")
    await msg.send()
    
    try:
        # Set up session with retry capability
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=3)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        # Choose API endpoint based on parallel setting
        api_url = SK_API_URL_PARALLEL if settings.parallel else SK_API_URL
        
        # Create step for API call with detailed information
        async with cl.Step(name="API Request", type="run") as step:
            step.input = {
                "endpoint": api_url,
                "query_rewrite": settings.query_rewrite,
                "planning": settings.planning,
                "web_search": settings.web_search,
                "ytb_search": settings.ytb_search,
                "mcp_server": settings.mcp_server,
                "search_engine": settings.search_engine,
                "parallel": settings.parallel,
                "verbose": settings.verbose,
                "locale": settings.language,
            }
            
            # Make request with stream=True
            response = session.post(
                api_url,
                json=payload,
                timeout=(5, 120),
                stream=True,
                headers={"Accept": "text/event-stream"}
            )
            
            step.output = f"Response status: {response.status_code}"
            
            logger.info(f"Response status: {response.status_code}, Content-Type: {response.headers.get('Content-Type', 'unknown')}")
            
            if response.status_code == 200:
                content_type = response.headers.get('Content-Type', '')
                
                if 'text/event-stream' in content_type:
                    # Process Server-Sent Events (SSE) with tool calling steps
                    async with cl.Step(name="Processing Response", type="tool") as process_step:
                        process_step.input = "Processing streaming response..."
                        
                        accumulated_content = ""
                        current_tool_step = None
                        tool_steps = {}
                        
                        logger.info("Starting SSE processing loop...")
                        for line in response.iter_lines():
                            if not line:
                                continue
                            
                            # Decode the line
                            line = line.decode('utf-8')
                            logger.info(f"SSE line received: {line}")
                            
                            # Skip SSE comments and empty lines
                            if line.startswith(':') or not line.strip():
                                continue
                            
                            # Handle SSE format (data: prefix)
                            if line.startswith('data: '):
                                line = line[6:].strip()  # Remove the 'data: ' prefix
                                
                                # Status message handling - create tool steps for different operations
                                if line.startswith('### '):
                                    step_content = line[4:]
                                    
                                    # Complete previous step if exists
                                    if current_tool_step:
                                        current_tool_step.output = "✅ Completed"
                                        await safe_send_step(current_tool_step)
                                    
                                    # Decode step content (name, code, description)
                                    step_name, code_content, description = decode_step_content(step_content)
                                    
                                    # Create new step for each tool operation with appropriate types
                                    step_type = "tool"
                                    step_icon = "🔧"
                                    
                                    # Determine step type and icon based on step name
                                    step_name_lower = step_name.lower()
                                    try:
                                        if ui_text.get("analyzing", "").lower() in step_name_lower:
                                            step_type = "intent"
                                            step_icon = "🧠"
                                        elif ui_text.get("analyze_complete", "").lower() in step_name_lower:
                                            step_type = "intent"
                                            step_icon = "🧠"
                                        elif ui_text.get("search_planning", "").lower() in step_name_lower:
                                            step_type = "planning"
                                            step_icon = "📋"
                                        elif ui_text.get("plan_done", "").lower() in step_name_lower:
                                            step_type = "planning"
                                            step_icon = "📋"
                                        elif ui_text.get("searching", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🌐"
                                        elif ui_text.get("search_done", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🌐"                                            
                                        elif ui_text.get("searching_YouTube", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🎬"
                                        elif ui_text.get("YouTube_done", "").lower() in step_name_lower:
                                            step_type = "retrieval"
                                            step_icon = "🎬"                                            
                                        elif ui_text.get("answering", "").lower() in step_name_lower:
                                            step_type = "llm"
                                            step_icon = "✏️"
                                        elif ui_text.get("search_and_answer", "").lower() in step_name_lower:
                                            step_type = "llm"
                                            step_icon = "✏️"
                                        elif "context information" in step_name_lower:
                                            step_type = "tool"
                                            step_icon = "📃"
                                    except KeyError as e:
                                        logger.warning(f"Missing UI text key: {e}")
                                    
                                    current_tool_step = cl.Step(
                                        name=f"{step_icon} {step_name}", 
                                        type=step_type
                                    )
                                    
                                    # Set input based on available content
                                    if code_content:
                                        # Display code with syntax highlighting
                                        current_tool_step.input = f"```python\n{code_content}\n```"
                                    elif description:
                                        # Display description
                                        current_tool_step.input = description
                                    else:
                                        # Default message
                                        current_tool_step.input = f"Executing: {step_name}"
                                    
                                    if not await safe_send_step(current_tool_step):
                                        logger.warning(f"Failed to send tool step: {step_name}")
                                        break  # Exit if connection is lost
                                    
                                    # Store step for later reference
                                    tool_steps[step_name] = current_tool_step
                            else:
                                # Regular content - accumulate and stream
                                if accumulated_content:
                                    # Apply formatting rules for line breaks
                                    if line.startswith(('•', '-', '#', '1.', '2.', '3.')) or accumulated_content.endswith(('.', '!', '?', ':')):
                                        accumulated_content += "\n\n" + line
                                    else:
                                        accumulated_content += "\n" + line
                                else:
                                    accumulated_content = line
                                
                                # Stream update to UI safely
                                if not await safe_stream_token(msg, line + "\n"):
                                    logger.warning("Stream connection lost, stopping streaming")
                                    break  # Exit if connection is lost
                        
                        # Close any remaining tool step
                        if current_tool_step:
                            current_tool_step.output = "✅ Completed"
                            await safe_send_step(current_tool_step)
                        
                        process_step.output = f"✅ Processed {len(accumulated_content)} characters across {len(tool_steps)} tool steps"
                
                else:
                    # Handle regular non-streaming response
                    async with cl.Step(name="Processing Non-Streaming Response", type="tool") as process_step:
                        logger.info("Not a chunked response, trying to process as regular response")
                        try:
                            chunks = []
                            for chunk in response.iter_content(chunk_size=None):
                                if chunk:
                                    chunks.append(chunk)
                            
                            if chunks:
                                response_text = b''.join(chunks).decode('utf-8', errors='replace')
                                
                                # Try to parse as JSON first
                                try:
                                    response_data = json.loads(response_text)
                                    if isinstance(response_data, dict) and "content" in response_data:
                                        await safe_stream_token(msg, response_data["content"])
                                        process_step.output = f"✅ Parsed JSON response with content: {response_data['content'][:50]}..."
                                    else:
                                        await safe_stream_token(msg, response_text)
                                        process_step.output = "✅ JSON response without content field, using raw text"
                                except json.JSONDecodeError:
                                    # Not valid JSON, just use as text
                                    await safe_stream_token(msg, response_text)
                                    process_step.output = "✅ Not a valid JSON response, using raw text"
                            else:
                                error_msg = "No response received from server."
                                await safe_stream_token(msg, error_msg)
                                process_step.output = error_msg
                        
                        except Exception as e:
                            error_msg = f"Error processing response: {str(e)}"
                            await safe_stream_token(msg, error_msg)
                            process_step.output = error_msg
            else:
                error_msg = f"Error: {response.status_code} - {response.text}"
                await safe_stream_token(msg, error_msg)
                step.output = error_msg
    
    except requests.exceptions.Timeout:
        error_msg = "Error: Request timed out. The server took too long to respond."
        await safe_stream_token(msg, error_msg)
        logger.error("Request timed out")
    except requests.exceptions.ConnectionError:
        error_msg = "Error: Connection failed. Please check if the API server is running."
        await safe_stream_token(msg, error_msg)
        logger.error("Connection error")
    except requests.exceptions.ChunkedEncodingError:
        error_msg = "Error: Connection interrupted while receiving data from the server."
        await safe_stream_token(msg, error_msg)
        logger.error("Chunked encoding error - connection interrupted")
    except requests.exceptions.RequestException as e:
        error_msg = f"Error connecting to the API: {str(e)}"
        await safe_stream_token(msg, error_msg)
        logger.error(f"Request exception: {type(e).__name__}: {str(e)}")
    except json.JSONDecodeError as e:
        error_msg = "Error: Received invalid JSON from the server."
        await safe_stream_token(msg, error_msg)
        logger.error(f"JSON decode error: {e}")
    except Exception as e:
        error_msg = f"Error: {str(e)}"
        await safe_stream_token(msg, error_msg)
        logger.error(f"Unexpected error in stream_chat_with_api: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Finalize the message safely
    await safe_update_message(msg)
    logger.info("Streaming completed")

@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages"""
    settings = cl.user_session.get("settings")
    if not settings:
        settings = ChatSettings()
        cl.user_session.set("settings", settings)
    
    # Process the message with streaming
    await stream_chat_with_api(message.content, settings)

@cl.action_callback("clear_chat")
async def on_action(action: cl.Action):
    """Handle clear chat action"""
    # Clear the chat context
    cl.chat_context.clear()
    
    # Send confirmation
    await cl.Message(content="Chat history cleared!").send()
    
    # Return success
    return "Chat cleared successfully"

@cl.action_callback("show_starters_action")
async def on_show_starters_action(action: cl.Action):
    """Handle show starters action"""
    current_profile = cl.user_session.get("chat_profile", "Korean")
    language = "ko-KR" if current_profile == "Korean" else "en-US"
    starters = get_starters_for_language(language)
    
    # Send starters as a message with action buttons
    starters_message = "📋 **Quick Start Options:**\n\n"
    actions = []
    
    for i, starter in enumerate(starters):
        # Get emoji from category mapping
        if i == 0:  # question_Microsoft
            emoji = "📈"
        elif i == 1:  # product_info
            emoji = "✅"
        elif i == 2:  # recommendation
            emoji = "💡"
        elif i == 3:  # comparison
            emoji = "📚"
        elif i == 4:  # support_questions
            emoji = "👨‍💻"
        elif i == 5:  # tools
            emoji = "🛠️"
        else:
            emoji = "🤖"
            
        starters_message += f"{emoji} **{starter.label}**\n"
        actions.append(
            cl.Action(
                name=f"starter_{i}",
                payload={"message": starter.message, "label": starter.label},
                label=f"{emoji} {starter.label}",
                description=f"Use starter: {starter.label}"
            )
        )
    
    await cl.Message(content=starters_message, actions=actions).send()
    return "Starters displayed"

@cl.action_callback("starter_0")
@cl.action_callback("starter_1")
@cl.action_callback("starter_2")
@cl.action_callback("starter_3")
@cl.action_callback("starter_4")
@cl.action_callback("starter_5")
async def on_starter_action(action: cl.Action):
    """Handle starter action clicks"""
    # Extract message from payload dictionary
    message_content = action.payload.get("message", "")
    starter_label = action.payload.get("label", "Unknown")
    
    logger.info(f"🎯 Starter action triggered: {action.name}")
    logger.info(f"📝 Message content: {message_content[:100]}...")
    logger.info(f"🏷️ Starter label: {starter_label}")
    
    # First, add the user message to chat history
    user_message = cl.Message(
        author="User",
        content=message_content,
        type="user_message"
    )
    await user_message.send()
    
    # Get current settings
    settings = cl.user_session.get("settings")
    if not settings:
        settings = ChatSettings()
        cl.user_session.set("settings", settings)
    
    # Process the starter message
    await stream_chat_with_api(message_content, settings)
    
    return f"Processing starter: {starter_label}"

if __name__ == "__main__":
    cl.run()
