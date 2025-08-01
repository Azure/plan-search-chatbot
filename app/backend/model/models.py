from typing import List, Optional
from pydantic import BaseModel, Field
from utils.enum import SearchEngine


class ChatMessage(BaseModel):
    """Chat message model with role and content"""
    role: str = Field(..., description="Role of the message sender (system, user, assistant)")
    content: str = Field(..., description="Content of the message")


class ChatRequest(BaseModel):
    """Chat request model with messages and optional parameters"""
    messages: List[ChatMessage] = Field(
        ..., 
        description="List of chat messages in the conversation history"
    )
    max_tokens: Optional[int] = Field(
        None, 
        description="Maximum number of tokens to generate"
    )
    temperature: Optional[float] = Field(
        None, 
        description="Temperature for response generation (0.0 to 1.0)"
    )
    stream: bool = Field(
        False,
        description="Enable streaming response"
    )
    query_rewrite: bool = Field(
        True,
        description="Enable query rewriting for better search results"
    )
    plan_execute: bool = Field(
        False,
        description="Enable plan and execute mode for complex queries"
    )
    search_engine: str = Field(
        SearchEngine.GOOGLE_SEARCH_CRAWLING,
        description="Search engine to use for retrieving information"
    )
    locale: str = Field(
        "ko-KR",
        description="Locale for the response"
    )


class ChatResponse(BaseModel):
    """Chat response model with message and success status"""
    message: str = Field(..., description="Response message from the chatbot")
    success: bool = Field(..., description="Indicates if the request was successful")
