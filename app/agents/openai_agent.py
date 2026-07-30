import asyncio
from typing import AsyncGenerator
from agents import Runner, set_default_openai_key
from app.interfaces.agent import BaseAgent
from app.dto.context import ExecutionContext
from app.dto.responses import AgentResponse
from app.services.tool_executor import ToolExecutor
from app.reasoning.reasoning_service import ReasoningService
from app.agents.agent_builder import AgentBuilder
from app.agents.mock_agent_fallback import MockAgentFallback
from app.config.settings import settings
from app.utils.logger import logger

class OpenAIAgent(BaseAgent):
    """
    Production Multi-Agent Architecture Service using OpenAI Agents SDK.
    Implements BaseAgent abstract interface (Dependency Inversion Principle).
    Delegates agent construction to AgentBuilder and offline fallback execution to MockAgentFallback.
    """

    def __init__(self, tool_executor: ToolExecutor | None = None, reasoning_service: ReasoningService | None = None):
        self.tool_executor = tool_executor or ToolExecutor()
        self.reasoning_service = reasoning_service or ReasoningService()
        self.agent_builder = AgentBuilder(self.tool_executor, self.reasoning_service)
        self.mock_fallback = MockAgentFallback(self.tool_executor, self.reasoning_service)
        
        if settings.agent.openai_api_key:
            set_default_openai_key(settings.agent.openai_api_key)

    async def run(self, system_prompt: str, user_request: str, context: ExecutionContext) -> AgentResponse:
        """Executes the dynamic multi-agent mesh using OpenAI Agents SDK Runner."""
        logger.info(f"Invoking Dynamic Multi-Agent SDK Runner with model {settings.agent.model}", component="OpenAIAgent", session_id=context.session_id)

        try:
            if settings.agent.openai_api_key and settings.agent.openai_api_key != "mock-key":
                triage_agent = self.agent_builder.build_team(context, system_prompt)
                result = await Runner.run(triage_agent, input=user_request)

                if hasattr(result.final_output, "model_dump_json"):
                    output_content = result.final_output.model_dump_json(indent=2)
                else:
                    output_content = str(result.final_output)

                context.agent_raw_response = output_content
                executed_calls = [
                    {"id": f"call-{res.tool_call_id}", "name": res.tool_name, "arguments": res.output}
                    for res in context.tool_results
                ]
                return AgentResponse(content=output_content, tool_calls=executed_calls)

        except Exception as e:
            logger.warning(f"OpenAI Multi-Agent SDK execution failed or in mock mode ({str(e)}). Delegating to MockAgentFallback pipeline.", component="OpenAIAgent")

        # Delegate to isolated mock fallback pipeline when OPENAI_API_KEY is unconfigured or call fails
        return await self.mock_fallback.run(system_prompt, user_request, context)

    async def run_stream(self, system_prompt: str, user_request: str, context: ExecutionContext) -> AsyncGenerator[str, None]:
        res = await self.run(system_prompt, user_request, context)
        words = res.content.split()
        for i in range(0, len(words), 3):
            chunk = " ".join(words[i:i+3]) + " "
            yield chunk
            await asyncio.sleep(0.03)
