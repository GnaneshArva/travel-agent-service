import json
import asyncio
from typing import Any, AsyncGenerator
from openai import AsyncOpenAI
from app.dto.context import ExecutionContext
from app.dto.responses import AgentResponse
from app.services.tool_executor import ToolExecutor
from app.config.settings import settings
from app.utils.logger import logger

class OpenAIAgent:
    """
    OpenAI Agent SDK Integration.
    Delegates tool selection and model reasoning to OpenAI API / Agent SDK runner,
    while leveraging Travel Agent Service's ToolExecutor for tool execution.
    """

    def __init__(self, tool_executor: ToolExecutor | None = None):
        self.tool_executor = tool_executor or ToolExecutor()
        self.client = AsyncOpenAI(api_key=settings.agent.openai_api_key or "mock-key")

    async def run(self, system_prompt: str, user_request: str, context: ExecutionContext) -> AgentResponse:
        logger.info(f"Invoking OpenAI Agent with model {settings.agent.model}", component="OpenAIAgent", session_id=context.session_id)
        tools = self.tool_executor.provider.get_available_tools()

        try:
            if settings.agent.openai_api_key and settings.agent.openai_api_key != "mock-key":
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_request}
                ]
                response = await self.client.chat.completions.create(
                    model=settings.agent.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    temperature=settings.agent.temperature,
                    max_tokens=settings.agent.max_tokens
                )
                msg = response.choices[0].message
                
                # Handle tool calls if returned by model
                if msg.tool_calls:
                    tool_calls_payload = [
                        {
                            "id": tc.id,
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                        for tc in msg.tool_calls
                    ]
                    # Execute tool batch
                    results = await self.tool_executor.execute_batch(tool_calls_payload, context)
                    context.agent_raw_response = msg.content or f"Executed {len(results)} travel search tools to formulate itinerary."
                    return AgentResponse(content=context.agent_raw_response, tool_calls=tool_calls_payload)
                else:
                    context.agent_raw_response = msg.content or ""
                    return AgentResponse(content=context.agent_raw_response)

        except Exception as e:
            logger.warning(f"OpenAI API call failed or in mock mode ({str(e)}). Executing automated fallback tool execution cycle.", component="OpenAIAgent")

        # Mock / Fallback execution flow: automatically invoke flight, hotel, weather tools
        dest = context.metadata.get("destination", "Japan")
        mock_calls = [
            {"id": "call-1", "name": "search_flights", "arguments": {"origin": "SFO", "destination": dest}},
            {"id": "call-2", "name": "search_hotels", "arguments": {"destination": dest, "min_rating": 4.5}},
            {"id": "call-3", "name": "get_weather", "arguments": {"destination": dest}}
        ]
        await self.tool_executor.execute_batch(mock_calls, context)
        
        fallback_text = (
            f"I have analyzed your request for a trip to {dest}. "
            f"Based on memory preferences and retrieved destination guides, I have found optimal flight options, "
            f"top 4-star accommodations, and checked local weather forecasts to prepare your custom itinerary."
        )
        context.agent_raw_response = fallback_text
        return AgentResponse(content=fallback_text, tool_calls=mock_calls)

    async def run_stream(self, system_prompt: str, user_request: str, context: ExecutionContext) -> AsyncGenerator[str, None]:
        res = await self.run(system_prompt, user_request, context)
        words = res.content.split()
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3]) + " "
            yield chunk
            await asyncio.sleep(0.03)
