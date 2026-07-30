# Step-by-Step Execution Architecture (`travel-agent-service`)

## Purpose
`travel-agent-service` is the central orchestration microservice for the Enterprise AI Travel Platform. Designed with Clean Architecture and SOLID principles, it coordinates session state, memory retrieval, knowledge RAG, prompt management, security guardrails, execution planning, MCP tool execution, and observability telemetry.

---

## Step-by-Step Request Execution Flow

```
Client Request
      │
      ▼
FastAPI TravelController
      │
      ▼
AgentOrchestrator
      │
 ┌────┴───────────────────────────┬───────────────────────────┬───────────────────────────┐
 ▼                                ▼                           ▼                           ▼
1. Session Init          2. Input Guardrails        3. Memory & Knowledge       4. Build Context
 (SessionManager)        (agentic-ai-guardrails)     (MCP Servers)               (ContextBuilder)
                                                                                          │
 ┌────────────────────────────────────────────────────────────────────────────────────────┘
 ▼
5. Execution Plan        6. Render Prompt           7. OpenAI Agent Loop        8. Tool Executor
 (PlanningService)       (agentic-ai-prompt-mgmt)    (ReasoningService)          (travel-mcp-server)
                                                                                          │
 ┌────────────────────────────────────────────────────────────────────────────────────────┘
 ▼
9. Output Guardrails     10. Response Processor     11. Observability           12. HTTP Response
 (agentic-ai-guardrails)  (Structured Strategy)      (agentic-ai-observability)   to Client
```

### Execution Lifecycle:
1. **Client HTTP Request**: Request received at `POST /api/v1/travel/plan` or `/api/v1/travel/plan/stream`.
2. **Session Initialization**: `SessionManager` instantiates `ExecutionContext` with session ID, conversation ID, request ID, and correlation trace ID.
3. **Input Guardrails**: `PlatformFacade` calls `agentic-ai-guardrails` to validate prompt safety.
4. **Prompt Template Loading**: `PlatformFacade` requests versioned system prompt templates from `agentic-ai-prompt-management`.
5. **Memory Retrieval**: User travel profile, seat preferences, and past trip history fetched from `travel-memory-mcp-server`.
6. **Knowledge Retrieval**: Destination guides, advisories, visa requirements, and weather retrieved from `travel-knowledge-mcp-server`.
7. **Context Construction**: `ContextBuilder` sequences context following strictly ordered composition rules.
8. **Execution Planning**: `PlanningService` determines step-by-step tool dependencies without executing them directly.
9. **OpenAI Agent Reasoning**: OpenAI model reasons over prompt, context, and available MCP tools.
10. **Parallel Tool Execution**: `ToolExecutor` executes MCP tools concurrently via `asyncio.gather()` over `travel-mcp-server`.
11. **Response Processing & Output Guardrails**: `ResponseProcessor` validates DTOs and verifies output against safety policies (RAG grounding, citations, business rules).
12. **Observability Telemetry**: Execution metrics, token usage, latency, and costs published to `agentic-ai-observability`.
