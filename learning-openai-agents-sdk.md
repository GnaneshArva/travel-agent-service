# Learning OpenAI Agents SDK: Architecture & Implementation Guide

This guide is an end-to-end masterclass on **OpenAI Agents SDK (`agents`)** based on the production implementation inside [`travel-agent-service`](file:///Users/gnanesh_arva/Downloads/travel-planner-v2/travel-agent-service).

---

## 1. What is the OpenAI Agents SDK?

The **OpenAI Agents SDK** (`agents`) is a lightweight, Python-native framework designed for building multi-agent AI applications. Unlike complex DAG frameworks, OpenAI Agents SDK centers around 3 core primitives:

1. **`Agent`**: Autonomous units of reasoning with dedicated system instructions, models, tool bindings, structured output schemas, and allowed handoffs.
2. **`Runner`**: The execution engine that orchestrates the event loop, tool calls, and handoff transitions.
3. **`Handoffs`**: Dynamic transfers of context and control between agents based on domain specialization.

---

## 2. Core Primitives Implemented in `travel-agent-service`

### A. Constructing an Agent (`Agent`)
Defined in [`app/agents/agent_builder.py`](file:///Users/gnanesh_arva/Downloads/travel-planner-v2/travel-agent-service/app/agents/agent_builder.py#L55-L119):

```python
from agents import Agent

flight_agent = Agent(
    name="FlightBookingAgent",
    instructions=(
        "You are the Flight Specialist. "
        "Use `search_flights` to find suitable flights to the destination. "
        "Hand off to HotelBookingAgent to book accommodations."
    ),
    model="gpt-4o",
    tools=[search_flights],
    output_type=FlightAgentOutput
)
```

- **`name`**: Identifier used in telemetry and agent handoff logs.
- **`instructions`**: System prompt defining role boundaries and operational rules.
- **`tools`**: Python functions wrapped with `@function_tool` decorator.
- **`output_type`**: Pydantic model enforcing structured JSON responses.

---

### B. Binding Tools (`@function_tool`)
Defined in [`app/agents/agent_builder.py`](file:///Users/gnanesh_arva/Downloads/travel-planner-v2/travel-agent-service/app/agents/agent_builder.py#L29-L48):

```python
from agents import function_tool

@function_tool
async def search_flights(origin: str, destination: str) -> str:
    """Flight Specialist tool to search available flight schedules and prices."""
    res = await tool_executor.execute_tool("search_flights", {"origin": origin, "destination": destination}, context)
    return json.dumps(res.output)
```

- Docstrings act as tool descriptions for the LLM.
- Argument type hints (`origin: str`, `destination: str`) generate the JSON Schema parameters automatically.

---

### C. Multi-Agent Handoff Mesh (`handoffs`)
Defined in [`app/agents/agent_builder.py`](file:///Users/gnanesh_arva/Downloads/travel-planner-v2/travel-agent-service/app/agents/agent_builder.py#L121-L125):

```python
# Wire multi-directional dynamic handoff mesh
triage_agent.handoffs = [flight_agent, hotel_agent]
flight_agent.handoffs = [hotel_agent, triage_agent]
hotel_agent.handoffs = [weather_agent, flight_agent, triage_agent]
weather_agent.handoffs = [synthesizer_agent, hotel_agent]
```

- **Dynamic Handoffs**: Allow agents to pass execution to each other based on intent and constraints (e.g. if hotel costs exceed budget, `HotelBookingAgent` hands back to `FlightBookingAgent` to check alternate cities).

---

### D. Executing the Mesh (`Runner.run`)
Defined in [`app/agents/openai_agent.py`](file:///Users/gnanesh_arva/Downloads/travel-planner-v2/travel-agent-service/app/agents/openai_agent.py#L37):

```python
from agents import Runner, set_default_openai_key

set_default_openai_key(settings.agent.openai_api_key)
result = await Runner.run(triage_agent, input=user_request)

# Extract structured output or string content
output_content = result.final_output.model_dump_json(indent=2)
```

- `Runner.run` starts execution with the entry agent (`triage_agent`), automatically manages tool executions, processes handoffs, and returns when the final agent emits its output.

---

## 3. Multi-Agent Request Lifecycle Diagram

```
User Request ──► Runner.run(triage_agent)
                        │
                        ▼
               TravelTriageAgent (Entry Router)
                        │
             ┌──────────┴──────────┐
             ▼                     ▼
   FlightBookingAgent    HotelBookingAgent
     (search_flights)      (search_hotels)
             │                     │
             └──────────┬──────────┘
                        ▼
              WeatherActivityAgent
                 (get_weather)
                        │
                        ▼
           ItinerarySynthesizerAgent
         (Final Structured Synthesis)
```

---

## 4. Features & Advanced Capabilities to Enhance the Service

While `travel-agent-service` currently uses agents, tools, handoffs, and structured outputs, the OpenAI Agents SDK offers several advanced features that can be added:

| Advanced SDK Feature | Description | Status in Current Project |
|---|---|---|
| **Guardrails & Input Hooks** | Attaching `@input_guardrail` to agents directly for inline validation. | Performed via HTTP facade (`agentic-ai-guardrails`). |
| **Agent State & Context Variables** | Passing custom state context objects into `@function_tool` functions via SDK context parameter. | Handled via custom `ExecutionContext` class in DI. |
| **Streaming Runner Events** | Consuming fine-grained events (`on_agent_change`, `on_tool_call`) from `Runner.run_stream()`. | Currently simulated word streaming. |
| **Tracing & OpenTelemetry SDK Hooks** | Built-in SDK telemetry integration with OpenTelemetry exporters. | Handled via HTTP facade (`agentic-ai-observability`). |
| **Dynamic Handoff Filters** | Conditional handoffs based on runtime evaluation predicates. | Static list `[flight_agent, hotel_agent]`. |

---

## 5. Summary & Code Reference Quick Links

- **Agent Builder Factory**: [`app/agents/agent_builder.py`](file:///Users/gnanesh_arva/Downloads/travel-planner-v2/travel-agent-service/app/agents/agent_builder.py)
- **OpenAI Agent Service**: [`app/agents/openai_agent.py`](file:///Users/gnanesh_arva/Downloads/travel-planner-v2/travel-agent-service/app/agents/openai_agent.py)
- **Agent Output DTO Schemas**: [`app/dto/agent_schemas.py`](file:///Users/gnanesh_arva/Downloads/travel-planner-v2/travel-agent-service/app/dto/agent_schemas.py)
