# Step-by-Step Execution Architecture (`travel-agent-service`)

## Purpose
`travel-agent-service` is the central orchestration microservice for the Enterprise AI Travel Platform. Designed with Clean Architecture and SOLID principles, it coordinates session state, memory retrieval, knowledge RAG, prompt management, security guardrails, execution planning, MCP tool execution, evaluation tracing, and observability telemetry.

---

## Step-by-Step Request Execution Flow

```
Client / Evals Request
      │
      ▼
FastAPI TravelController / EvalController
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
9. Output Guardrails     10. Automated Remediation  11. Response Processor      12. Telemetry & Output
 (agentic-ai-guardrails)  (Multi-Violation Loop)     (Structured Strategy)       (agentic-ai-obs / Client)
```

### Execution Lifecycle:
1. **Client HTTP Request**: Request received at `POST /api/v1/travel/plan`, `POST /api/v1/travel/plan/stream`, or `POST /api/v1/travel/evaluate` (from `agentic-ai-evals`).
2. **Session Initialization**: `SessionManager` instantiates `ExecutionContext` with session ID, conversation ID, request ID, and correlation trace ID.
3. **Input Guardrails**: `PlatformFacade` calls `agentic-ai-guardrails` (`POST /guardrails/input/validate`) to validate prompt safety. (Note: Initial perimeter checks are performed upstream at `agentic-ai-gateway`).
4. **Prompt Template Loading**: `PlatformFacade` requests versioned system prompt templates from `agentic-ai-prompt-management`.
5. **Memory Retrieval**: User travel profile, seat preferences, and past trip history fetched from `travel-memory-mcp-server`.
6. **Knowledge Retrieval**: Destination guides, advisories, visa requirements, and weather retrieved from `travel-knowledge-mcp-server`.
7. **Context Construction**: `ContextBuilder` sequences context following strictly ordered composition rules.
8. **Execution Planning**: `PlanningService` determines step-by-step tool dependencies without executing them directly.
9. **OpenAI Agent Reasoning**: OpenAI model reasons over prompt, context, and available MCP tools.
10. **Parallel Tool Execution**: `ToolExecutor` executes MCP tools concurrently via `asyncio.gather()` over `travel-mcp-server`.
11. **Output Guardrails**: `ResponseProcessor` calls `agentic-ai-guardrails` (`POST /guardrails/output/validate`) to perform RAG grounding checks, citation verification, and business rule enforcement.
12. **Automated Multi-Violation Remediation**:
    - If 1–2 recoverable violations occur $\rightarrow$ automatically formats violations into a system instruction and re-prompts the agent (max 2 retries).
    - If $\ge 3$ violations or `CRITICAL` severity $\rightarrow$ immediately triggers circuit breaker and returns a deterministic safe fallback response.
13. **Response Processing**: `ResponseProcessor` formats final DTO response payload (`TravelResponse` or `EvalTraceResponse`).
14. **Observability Telemetry**: Execution metrics, token usage, latency, and costs published to `agentic-ai-observability`.
