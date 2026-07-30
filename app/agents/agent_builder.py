import json
from agents import Agent, function_tool
from app.dto.context import ExecutionContext
from app.dto.agent_schemas import (
    TriageAgentOutput,
    FlightAgentOutput,
    HotelAgentOutput,
    WeatherAgentOutput,
    FinalItinerarySynthesisOutput
)
from app.services.tool_executor import ToolExecutor
from app.reasoning.reasoning_service import ReasoningService
from app.config.settings import settings

class AgentBuilder:
    """
    Dedicated Multi-Agent Team Factory.
    Constructs domain-specialized agents with tool bindings, instructions, and dynamic handoff mesh.
    """

    def __init__(self, tool_executor: ToolExecutor | None = None, reasoning_service: ReasoningService | None = None):
        self.tool_executor = tool_executor or ToolExecutor()
        self.reasoning_service = reasoning_service or ReasoningService()

    def build_team(self, context: ExecutionContext, system_prompt: str) -> Agent:
        """Constructs a network of specialized agents with dynamic multi-directional handoff mesh."""

        # Function tools bound to domain specialists
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

        # ------------------------------------------------------------------
        # DYNAMIC MULTI-AGENT DEFINITIONS & BUDGET REASONING
        # ------------------------------------------------------------------

        # 1. TravelTriageAgent (Primary Entry Router)
        triage_agent = Agent(
            name="TravelTriageAgent",
            instructions=(
                f"{system_prompt}\n\n"
                "You are the Travel Triage Orchestrator. "
                "Analyze user travel intent, budget, and constraints. "
                "Route request to FlightBookingAgent or HotelBookingAgent."
            ),
            model=settings.agent.model,
            output_type=TriageAgentOutput
        )

        # 2. FlightBookingAgent (Flight Specialist)
        flight_agent = Agent(
            name="FlightBookingAgent",
            instructions=(
                "You are the Flight Specialist. "
                "Use `search_flights` to find suitable flights to the destination. "
                "Hand off to HotelBookingAgent to book accommodations. "
                "If handed back a request to evaluate lower-cost alternate cities, search flights for the new location."
            ),
            model=settings.agent.model,
            tools=[search_flights],
            output_type=FlightAgentOutput
        )

        # 3. HotelBookingAgent (Lodging & Budget Specialist)
        hotel_agent = Agent(
            name="HotelBookingAgent",
            instructions=(
                "You are the Lodging & Budget Specialist. "
                "Use `search_hotels` to find accommodations matching user preferences. "
                "BUDGET OPTIMIZATION RULE: If hotel stay prices in the primary city exceed user budget, "
                "search lower-cost accommodations in nearby cities or hand off back to FlightBookingAgent "
                "to evaluate alternative flight connections before continuing to WeatherActivityAgent."
            ),
            model=settings.agent.model,
            tools=[search_hotels],
            output_type=HotelAgentOutput
        )

        # 4. WeatherActivityAgent (Weather & Activity Specialist)
        weather_agent = Agent(
            name="WeatherActivityAgent",
            instructions=(
                "You are the Weather and Activity Specialist. "
                "Use the `get_weather` tool to check forecast conditions for the destination. "
                "Once forecast is retrieved, hand off control to ItinerarySynthesizerAgent."
            ),
            model=settings.agent.model,
            tools=[get_weather],
            output_type=WeatherAgentOutput
        )

        # 5. ItinerarySynthesizerAgent (Final Synthesis Engine)
        synthesizer_agent = Agent(
            name="ItinerarySynthesizerAgent",
            instructions=(
                "You are the Final Itinerary Synthesizer Engine. "
                "Review all retrieved flight choices, lodging details, weather forecasts, and budget trade-offs, "
                "and produce a comprehensive, structured final travel plan synthesis matching the required schema."
            ),
            model=settings.agent.model,
            output_type=FinalItinerarySynthesisOutput
        )

        # Wire multi-directional dynamic handoff mesh (allows loops & re-routing)
        triage_agent.handoffs = [flight_agent, hotel_agent]
        flight_agent.handoffs = [hotel_agent, triage_agent]
        hotel_agent.handoffs = [weather_agent, flight_agent, triage_agent]
        weather_agent.handoffs = [synthesizer_agent, hotel_agent]

        return triage_agent
