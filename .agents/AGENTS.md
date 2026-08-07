# travel-agent-service Handbook

## Universal Rules
- **Git Push Approval Rule**: NEVER run `git push` automatically. Always present implemented changes and unit test verification results, and wait for explicit user confirmation before executing any `git push` command.
- **Python Virtualenv Path**: All unit tests must be executed using:
  `/Users/gnanesh_arva/Downloads/travel-planner-v2/travel-agent-service/.venv/bin/pytest`

## Repository Standards
- **Port**: `8000` (Default)
- **Role**: Central Multi-Agent Orchestration Service using OpenAI Agents SDK (`AgentBuilder` team mesh, prompt caching, HITL approval).

## Relevant Task Playbooks (`skills/`)
- `hitl-approval-workflow`: Human-in-the-Loop risk assessment and approval state resolution.
