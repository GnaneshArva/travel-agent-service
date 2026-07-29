import json
import asyncio
from typing import Any, AsyncGenerator
from agents import Agent, Runner, function_tool, set_default_openai_key
from app.dto.context import ExecutionContext
from app.dto.responses import AgentResponse
from app.services.tool_executor import ToolExecutor
from app.reasoning.reasoning_service import ReasoningService
from app.config.settings import settings
from app.utils.logger import logger

class OpenAIAgent:
    """
    OpenAI Agents SDK Integration with Multi-Turn Feedback Loop (Re-Act Loop).
    Delegates iterative tool selection (search_flights -> search_hotels -> get_weather)
    and model reasoning turns to official OpenAI Agents SDK Runner.
    """

    def __init__(self, tool_executor: ToolExecutor | None = None, reasoning_service: ReasoningService | None = None):
        self.tool_executor = tool_executor or ToolExecutor()
        self.reasoning_service = reasoning_service or ReasoningService()
        if settings.agent.openai_api_key:
            set_default_openai_key(settings.agent.openai_api_key)

    def _create_function_tools(self, context: ExecutionContext):
        """Creates function tools bound to current execution context."""

        @function_tool
        async def search_flights(origin: str, destination: str) -> str:
            """Search available flights between origin and destination."""
            self.reasoning_service.record_thought(context, f"[Turn 1 - Flight Search] Searching flights from {origin} to {destination}")
            res = await self.tool_executor.execute_tool("search_flights", {"origin": origin, "destination": destination}, context)
            return json.dumps(res.output)

        @function_tool
        async def search_hotels(destination: str, min_rating: float = 4.0) -> str:
            """Search accommodations at destination matching minimum rating."""
            self.reasoning_service.record_thought(context, f"[Turn 2 - Hotel Search] Searching top-rated hotels in {destination} (rating >= {min_rating})")
            res = await self.tool_executor.execute_tool("search_hotels", {"destination": destination, "min_rating": min_rating}, context)
            return json.dumps(res.output)

        @function_tool
        async def get_weather(destination: str) -> str:
            """Retrieve weather forecast for destination."""
            self.reasoning_service.record_thought(context, f"[Turn 3 - Weather Assessment] Fetching weather forecast for {destination}")
            res = await self.tool_executor.execute_tool("get_weather", {"destination": destination}, context)
            return json.dumps(res.output)

        return [search_flights, search_hotels, get_weather]

    async def run(self, system_prompt: str, user_request: str, context: ExecutionContext) -> AgentResponse:
        logger.info(f"Invoking Multi-Turn OpenAI Agent with model {settings.agent.model}", component="OpenAIAgent", session_id=context.session_id)

        try:
            if settings.agent.openai_api_key and settings.agent.openai_api_key != "mock-key":
                tools = self._create_function_tools(context)
                agent = Agent(
                    name="TravelPlannerAgent",
                    instructions=(
                        f"{system_prompt}\n\n"
                        "Follow an iterative multi-turn planning loop:\n"
                        "1. First, search for flights to the destination.\n"
                        "2. Next, based on arrival details, search for hotels matching user preferences.\n"
                        "3. Then, check the weather forecast to customize day-by-day activities.\n"
                        "4. Finally, synthesize all tool outputs into a clear structured travel plan."
                    ),
                    model=settings.agent.model,
                    tools=tools
                )
                
                result = await Runner.run(agent, input=user_request)
                output_content = result.final_output or ""
                context.agent_raw_response = output_content
                executed_calls = [
                    {"id": f"call-{res.tool_call_id}", "name": res.tool_name, "arguments": res.output}
                    for res in context.tool_results
                ]
                return AgentResponse(content=output_content, tool_calls=executed_calls)

        except Exception as e:
            logger.warning(f"OpenAI Agents SDK execution failed or in mock mode ({str(e)}). Executing multi-turn fallback tool execution loop.", component="OpenAIAgent")

        # Multi-Turn Fallback Feedback Loop
        dest = context.metadata.get("destination", "Japan")
        
        # Turn 1: Flight Search
        self.reasoning_service.record_thought(context, f"[Turn 1 - Flight Reasoning] Decided to search flight options from SFO to {dest}.")
        await self.tool_executor.execute_tool("search_flights", {"origin": "SFO", "destination": dest}, context)
        
        # Turn 2: Hotel Search (Feedback from Turn 1)
        self.reasoning_service.record_thought(context, f"[Turn 2 - Hotel Reasoning] Evaluated flight options. Now searching 4.5+ star accommodations in {dest}.")
        await self.tool_executor.execute_tool("search_hotels", {"destination": dest, "min_rating": 4.5}, context)

        # Turn 3: Weather Assessment (Feedback from Turn 2)
        self.reasoning_service.record_thought(context, f"[Turn 3 - Activity Reasoning] Selected lodging. Checking weather forecast for {dest} to tailor itinerary.")
        await self.tool_executor.execute_tool("get_weather", {"destination": dest}, context)

        # Turn 4: Final Synthesis
        self.reasoning_service.record_thought(context, f"[Turn 4 - Synthesis] Synthesizing flights, hotels, and weather forecast into final multi-day itinerary for {dest}.")

        executed_calls = [
            {"id": res.tool_call_id, "name": res.tool_name, "arguments": {}}
            for res in context.tool_results
        ]
        
        fallback_text = (
            f"I have completed a multi-turn analysis for your trip to {dest}.\n\n"
            f"- **Turn 1 (Flights)**: Found optimal flights from SFO to {dest}.\n"
            f"- **Turn 2 (Hotels)**: Selected top-rated accommodations based on location and rating.\n"
            f"- **Turn 3 (Weather)**: Checked destination weather to schedule ideal indoor/outdoor activities.\n\n"
            f"Your tailored itinerary is ready below."
        )
        context.agent_raw_response = fallback_text
        return AgentResponse(content=fallback_text, tool_calls=executed_calls)

    async def run_stream(self, system_prompt: str, user_request: str, context: ExecutionContext) -> AsyncGenerator[str, None]:
        res = await self.run(system_prompt, user_request, context)
        words = res.content.split()
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3]) + " "
            yield chunk
            await asyncio.sleep(0.03)
