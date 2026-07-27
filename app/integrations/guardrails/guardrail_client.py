import httpx
from typing import Any
from app.interfaces.guardrail import GuardrailProvider
from app.config.settings import settings
from app.utils.logger import logger

class GuardrailClient(GuardrailProvider):
    """Integration Client for agentic-ai-guardrails with standalone fallback."""

    def __init__(self, service_url: str | None = None):
        self.service_url = service_url or settings.mcp.guardrails_service_url
        self.timeout = settings.timeouts.mcp_timeout

    async def validate_input(self, text: str, session_id: str | None = None) -> tuple[bool, str, dict[str, Any]]:
        if not settings.features.enable_guardrails:
            return True, text, {}

        logger.info("Evaluating input guardrails", component="GuardrailClient", session_id=session_id)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.service_url}/guardrails/input/validate", json={"text": text, "session_id": session_id})
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("is_allowed", True), data.get("sanitized_text", text), data.get("metadata", {})
        except Exception:
            logger.info("Guardrails service unreachable. Permitting request via local verification.", component="GuardrailClient")

        # Fallback basic prompt injection check
        forbidden_terms = ["ignore previous instructions", "system prompt leak"]
        for term in forbidden_terms:
            if term in text.lower():
                return False, f"Input blocked due to policy term: {term}", {"reason": "prompt_injection"}

        return True, text, {"mode": "local_fallback"}

    async def validate_output(self, text: str, session_id: str | None = None) -> tuple[bool, str, dict[str, Any]]:
        if not settings.features.enable_guardrails:
            return True, text, {}

        logger.info("Evaluating output guardrails", component="GuardrailClient", session_id=session_id)
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.post(f"{self.service_url}/guardrails/output/validate", json={"text": text, "session_id": session_id})
                if resp.status_code == 200:
                    data = resp.json()
                    return data.get("is_allowed", True), data.get("sanitized_text", text), data.get("metadata", {})
        except Exception:
            pass

        return True, text, {"mode": "local_fallback"}
