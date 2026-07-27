import httpx
from typing import Any
from app.interfaces.prompt import PromptProvider
from app.dto.context import PromptContext
from app.config.settings import settings
from app.utils.logger import logger

class PromptManagementClient(PromptProvider):
    """Integration Client for agentic-ai-prompt-management with standalone fallback."""

    def __init__(self, service_url: str | None = None):
        self.service_url = service_url or settings.mcp.prompt_management_url
        self.timeout = settings.timeouts.mcp_timeout

    async def load_prompt(
        self,
        template_name: str,
        version: str | None = None,
        variables: dict[str, Any] | None = None
    ) -> PromptContext:
        vars_dict = variables or {}
        version_str = version or settings.prompt.prompt_version
        logger.info(f"Loading prompt template '{template_name}' version {version_str}", component="PromptManagementClient")

        if settings.features.enable_prompt_management:
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        f"{self.service_url}/prompts/render",
                        json={"template_name": template_name, "version": version_str, "variables": vars_dict}
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        return PromptContext(
                            template_name=template_name,
                            rendered_prompt=data.get("rendered_prompt", ""),
                            system_instruction=data.get("system_instruction", ""),
                            variables=vars_dict
                        )
            except Exception as e:
                logger.warning(f"Prompt Management service unreachable ({str(e)}). Falling back to default system template.", component="PromptManagementClient")

        default_system_prompt = (
            "You are an Enterprise AI Travel Planner orchestrator. "
            "Your role is to plan trips, coordinate flight and hotel searches, retrieve relevant travel advisories, "
            "and craft comprehensive, personalized travel itineraries using registered MCP tools."
        )
        return PromptContext(
            template_name=template_name,
            rendered_prompt=default_system_prompt,
            system_instruction=default_system_prompt,
            variables=vars_dict
        )
