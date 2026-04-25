from dataclasses import dataclass


@dataclass(frozen=True)
class RoleSpec:
    name: str
    summary: str
    system_prompt: str


ROLE_SPECS = {
    "coordinator": RoleSpec(
        name="coordinator",
        summary="Routes K8s benchmark work across planning, execution, and verification roles.",
        system_prompt=(
            "You are the coordinator for a custom purple-team Kubernetes benchmark agent. "
            "Turn the user's request into an actionable response. Prefer concrete kubectl, "
            "Kubernetes API, and incident-response steps. Be explicit about assumptions and "
            "verification."
        ),
    ),
    "planner": RoleSpec(
        name="planner",
        summary="Builds concise K8s investigation and remediation plans.",
        system_prompt=(
            "You are the planning subagent for a Kubernetes purple-team benchmark. "
            "Produce a compact plan with commands, expected observations, and fallback paths."
        ),
    ),
    "verifier": RoleSpec(
        name="verifier",
        summary="Checks whether proposed K8s actions are complete and safe enough to submit.",
        system_prompt=(
            "You are the verification subagent for a Kubernetes purple-team benchmark. "
            "Inspect the proposed response for missing checks, unsafe assumptions, and final "
            "evidence the coordinator should gather."
        ),
    ),
}


def get_role(role: str) -> RoleSpec:
    return ROLE_SPECS.get(role, ROLE_SPECS["coordinator"])
