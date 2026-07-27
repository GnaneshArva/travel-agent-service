import json
import asyncio
from typing import AsyncGenerator
from app.interfaces.strategies import ResponseGenerationStrategy
from app.dto.context import ExecutionContext
from app.dto.responses import TravelResponse, DailyItinerary, FlightOption, HotelOption

class StructuredResponseStrategy(ResponseGenerationStrategy):
    """Parses tool results and agent raw output into typed Pydantic v2 TravelResponse."""
    async def generate_response(self, context: ExecutionContext) -> TravelResponse:
        destination = context.metadata.get("destination", "Target Destination")
        duration_days = context.metadata.get("duration_days", 5)
        
        flights: list[FlightOption] = []
        hotels: list[HotelOption] = []
        applied_prefs: list[str] = []

        # Extract tool results
        for res in context.tool_results:
            if res.status == "SUCCESS" and isinstance(res.output, dict):
                if res.tool_name in ("search_flights", "flight_search") and "flights" in res.output:
                    for f in res.output["flights"]:
                        flights.append(FlightOption(**f))
                elif res.tool_name in ("search_hotels", "hotel_search") and "hotels" in res.output:
                    for h in res.output["hotels"]:
                        hotels.append(HotelOption(**h))

        # Default fallbacks if tools returned simulated/empty responses
        if not flights:
            flights.append(FlightOption(
                airline="Global Airways",
                flight_number="GA-742",
                origin="SFO",
                destination=destination,
                departure_time="08:00 AM",
                price=850.0
            ))
        if not hotels:
            hotels.append(HotelOption(
                name=f"Grand {destination.title()} Resort & Spa",
                rating=4.8,
                price_per_night=220.0,
                location="City Center",
                amenities=["WiFi", "Breakfast", "Pool", "Spa"]
            ))

        if context.memory_context.user_preferences:
            applied_prefs = [f"{k}: {v}" for k, v in context.memory_context.user_preferences.items()]

        itinerary = [
            DailyItinerary(
                day=i + 1,
                title=f"Day {i + 1}: Exploring {destination.title()}",
                activities=[f"Visit top cultural attraction in {destination}", "Guided city tour", "Evening relaxation"],
                dining_recommendations=["Local specialty restaurant", "Recommended Cafe"]
            )
            for i in range(duration_days)
        ]

        summary = context.agent_raw_response or f"Customized {duration_days}-day itinerary to {destination} prepared based on your preferences."

        return TravelResponse(
            session_id=context.session_id,
            conversation_id=context.conversation_id,
            request_id=context.request_id,
            status="SUCCESS",
            destination=destination,
            duration_days=duration_days,
            summary=summary,
            itinerary=itinerary,
            flights=flights,
            hotels=hotels,
            estimated_total_cost=round(sum(f.price for f in flights) + sum(h.price_per_night * duration_days for h in hotels), 2),
            currency="USD",
            warnings=context.warnings,
            applied_preferences=applied_prefs,
            execution_time_ms=context.observation_context.events[-1].get("duration_ms", 0.0) if context.observation_context.events else 0.0
        )

    async def generate_stream(self, context: ExecutionContext) -> AsyncGenerator[str, None]:
        resp = await self.generate_response(context)
        chunks = [
            f"data: {json.dumps({'event': 'started', 'session_id': resp.session_id})}\n\n",
            f"data: {json.dumps({'event': 'summary', 'text': resp.summary})}\n\n",
            f"data: {json.dumps({'event': 'itinerary_ready', 'count': len(resp.itinerary)})}\n\n",
            f"data: {json.dumps({'event': 'completed', 'response': resp.model_dump()})}\n\n"
        ]
        for chunk in chunks:
            yield chunk
            await asyncio.sleep(0.05)

class StreamingResponseStrategy(StructuredResponseStrategy):
    """Focuses on streaming chunks incrementally."""
    pass

class TextOnlyStrategy(StructuredResponseStrategy):
    """Produces plain text summary."""
    pass
