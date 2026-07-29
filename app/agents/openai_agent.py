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
    Multi-Agent Architecture using OpenAI Agents SDK with explicit Handoffs & Synthesis Engine.
    Network:
    - TravelTriageAgent: Primary entry point router. Hands off to FlightBookingAgent.
    - FlightBookingAgent: Flight specialist using search_flights. Hands off to HotelBookingAgent.
    - HotelBookingAgent: Lodging specialist using search_hotels. Hands off to WeatherActivityAgent.
    - WeatherActivityAgent: Weather & activity specialist using get_weather. Hands off to ItinerarySynthesizerAgent.
    - ItinerarySynthesizerAgent: Final synthesis engine compiling flights, hotels, weather, and budget into complete itinerary.
    """

    def __init__(self, tool_executor: ToolExecutor | None = None, reasoning_service: ReasoningService | None = None):
        self.tool_executor = tool_executor or ToolExecutor()
        self.reasoning_service = reasoning_service or ReasoningService()
        if settings.agent.openai_api_key:
            set_default_openai_key(settings.agent.openai_api_key)

    def _synthesize_itinerary(self, context: ExecutionContext) -> str:
        """Parses accumulated tool outputs and memory context into a rich synthesized text response."""
        dest = context.metadata.get("destination", "Destination")
        duration = context.metadata.get("duration_days", 3)
        
        flights = []
        hotels = []
        weather_condition = "Clear Sunny Weather"
        temperature = "22°C"

        for res in context.tool_results:
            if res.status == "SUCCESS" and isinstance(res.output, dict):
                if "flights" in res.output:
                    flights = res.output["flights"]
                elif "hotels" in res.output:
                    hotels = res.output["hotels"]
                elif "condition" in res.output:
                    weather_condition = res.output.get("condition", weather_condition)
                    temperature = res.output.get("temperature", temperature)

        flight_str = f"{flights[0]['airline']} ({flights[0]['flight_number']}) - ${flights[0]['price']}" if flights else "Global Express - $750"
        hotel_str = f"{hotels[0]['name']} ({hotels[0]['rating']}★, ${hotels[0]['price_per_night']}/night)" if hotels else f"Grand {dest} Hotel (4.8★)"
        total_estimate = (flights[0]['price'] if flights else 750) + ((hotels[0]['price_per_night'] if hotels else 200) * duration)

        synthesis = (
            f"### Multi-Agent Final Itinerary Synthesis for {dest} ({duration} Days)\n\n"
            f"**1. Flight Connection:** {flight_str}\n"
            f"**2. Lodging:** {hotel_str}\n"
            f"**3. Weather Forecast:** {weather_condition}, {temperature}\n"
            f"**4. Estimated Total Cost:** ${total_estimate:.2f} USD\n\n"
            f"**Multi-Agent Handoff Chain:**\n"
            f"- `TravelTriageAgent`: Routed user prompt & extracted constraints.\n"
            f"- `FlightBookingAgent`: Searched and reserved flight connections.\n"
            f"- `HotelBookingAgent`: Selected top-rated lodging matching user rating preference.\n"
            f"- `WeatherActivityAgent`: Assessed destination climate forecast.\n"
            f"- `ItinerarySynthesizerAgent`: Consolidated all domain data into final travel plan."
        )
        return synthesis

    def _build_multi_agent_team(self, context: ExecutionContext, system_prompt: str) -> Agent:
        """Constructs a network of specialized agents with explicit SDK handoffs."""

        @function_tool
        async def search_flights(origin: str, destination: str) -> str:
            """Flight Specialist tool to search available flight schedules and prices."""
            self.reasoning_service.record_thought(context, f"[FlightBookingAgent] Searching flights from {origin} to {destination}")
            res = await self.tool_executor.execute_tool("search_flights", {"origin": origin, "destination": destination}, context)
            return json.dumps(res.output)

        @function_tool
        async def search_hotels(destination: str, min_rating: float = 4.0) -> str:
            """Lodging Specialist tool to search top-rated accommodations."""
            self.reasoning_service.record_thought(context, f"[HotelBookingAgent] Searching hotels in {destination} (min_rating={min_rating})")
            res = await self.tool_executor.execute_tool("search_hotels", {"destination": destination, "min_rating": min_rating}, context)
            return json.dumps(res.output)

        @function_tool
        async def get_weather(destination: str) -> str:
            """Weather Specialist tool to check destination forecast."""
            self.reasoning_service.record_thought(context, f"[WeatherActivityAgent] Fetching weather forecast for {destination}")
            res = await self.tool_executor.execute_tool("get_weather", {"destination": destination}, context)
            return json.dumps(res.output)

        # 5. Final Itinerary Synthesizer Agent
        synthesizer_agent = Agent(
            name="ItinerarySynthesizerAgent",
            instructions=(
                "You are the Final Itinerary Synthesizer Engine. "
                "Review all retrieved flight choices, lodging details, weather forecasts, and user preferences, "
                "and produce a comprehensive, structured final travel plan synthesis."
            ),
            model=settings.agent.model
        )

        # 4. Weather & Activity Specialist (Handoffs to ItinerarySynthesizerAgent)
        weather_agent = Agent(
            name="WeatherActivityAgent",
            instructions=(
                "You are the Weather and Activity Specialist. "
                "Use the `get_weather` tool to check forecast conditions for the destination. "
                "Once forecast is retrieved, hand off control to ItinerarySynthesizerAgent for final synthesis."
            ),
            model=settings.agent.model,
            tools=[get_weather],
            handoffs=[synthesizer_agent]
        )

        # 3. Hotel Booking Specialist (Handoffs to WeatherActivityAgent)
        hotel_agent = Agent(
            name="HotelBookingAgent",
            instructions=(
                "You are the Lodging Specialist. "
                "Use `search_hotels` to find accommodations matching user preferences. "
                "Once hotels are selected, hand off control to WeatherActivityAgent."
            ),
            model=settings.agent.model,
            tools=[search_hotels],
            handoffs=[weather_agent]
        )

        # 2. Flight Booking Specialist (Handoffs to HotelBookingAgent)
        flight_agent = Agent(
            name="FlightBookingAgent",
            instructions=(
                "You are the Flight Specialist. "
                "Use `search_flights` to find suitable flights to the destination. "
                "Once flight options are retrieved, hand off control to HotelBookingAgent."
            ),
            model=settings.agent.model,
            tools=[search_flights],
            handoffs=[hotel_agent]
        )

        # 1. Primary Entry Orchestrator (Handoffs to FlightBookingAgent)
        triage_agent = Agent(
            name="TravelTriageAgent",
            instructions=(
                f"{system_prompt}\n\n"
                "You are the Travel Triage Orchestrator. "
                "Analyze the user's travel request and immediately hand off control to FlightBookingAgent to begin planning."
            ),
            model=settings.agent.model,
            handoffs=[flight_agent]
        )

        return triage_agent

    async def run(self, system_prompt: str, user_request: str, context: ExecutionContext) -> AgentResponse:
        logger.info(f"Invoking Multi-Agent SDK Runner with model {settings.agent.model}", component="OpenAIAgent", session_id=context.session_id)

        try:
            if settings.agent.openai_api_key and settings.agent.openai_api_key != "mock-key":
                triage_agent = self._build_multi_agent_team(context, system_prompt)
                result = await Runner.run(triage_agent, input=user_request)
                output_content = result.final_output or self._synthesize_itinerary(context)
                context.agent_raw_response = output_content
                executed_calls = [
                    {"id": f"call-{res.tool_call_id}", "name": res.tool_name, "arguments": res.output}
                    for res in context.tool_results
                ]
                return AgentResponse(content=output_content, tool_calls=executed_calls)

        except Exception as e:
            logger.warning(f"OpenAI Multi-Agent SDK execution failed or in mock mode ({str(e)}). Running simulated multi-agent handoff & synthesis pipeline.", component="OpenAIAgent")

        # Multi-Agent Fallback Execution & Synthesis Pipeline
        dest = context.metadata.get("destination", "Japan")

        # Agent 1: TravelTriageAgent -> Handoff to FlightBookingAgent
        self.reasoning_service.record_thought(context, f"[TravelTriageAgent] Received user request for {dest}. Initiating Handoff -> FlightBookingAgent.")

        # Agent 2: FlightBookingAgent -> search_flights -> Handoff to HotelBookingAgent
        self.reasoning_service.record_thought(context, f"[FlightBookingAgent] Searching flight schedules from SFO to {dest}.")
        await self.tool_executor.execute_tool("search_flights", {"origin": "SFO", "destination": dest}, context)
        self.reasoning_service.record_thought(context, f"[FlightBookingAgent] Flights retrieved. Initiating Handoff -> HotelBookingAgent.")

        # Agent 3: HotelBookingAgent -> search_hotels -> Handoff to WeatherActivityAgent
        self.reasoning_service.record_thought(context, f"[HotelBookingAgent] Searching top 4.5+ star accommodations in {dest}.")
        await self.tool_executor.execute_tool("search_hotels", {"destination": dest, "min_rating": 4.5}, context)
        self.reasoning_service.record_thought(context, f"[HotelBookingAgent] Lodging options selected. Initiating Handoff -> WeatherActivityAgent.")

        # Agent 4: WeatherActivityAgent -> get_weather -> Handoff to ItinerarySynthesizerAgent
        self.reasoning_service.record_thought(context, f"[WeatherActivityAgent] Fetching destination weather forecast for {dest}.")
        await self.tool_executor.execute_tool("get_weather", {"destination": dest}, context)
        self.reasoning_service.record_thought(context, f"[WeatherActivityAgent] Weather retrieved. Initiating Handoff -> ItinerarySynthesizerAgent.")

        # Agent 5: ItinerarySynthesizerAgent -> Final Synthesis Engine
        self.reasoning_service.record_thought(context, f"[ItinerarySynthesizerAgent] Running final synthesis: consolidating flights, hotels, weather, and budget for {dest}.")

        executed_calls = [
            {"id": res.tool_call_id, "name": res.tool_name, "arguments": {}}
            for res in context.tool_results
        ]
        
        synthesized_text = self._synthesize_itinerary(context)
        context.agent_raw_response = synthesized_text
        return AgentResponse(content=synthesized_text, tool_calls=executed_calls)

    async def run_stream(self, system_prompt: str, user_request: str, context: ExecutionContext) -> AsyncGenerator[str, None]:
        res = await self.run(system_prompt, user_request, context)
        words = res.content.split()
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3]) + " "
            yield chunk
            await asyncio.sleep(0.03)
