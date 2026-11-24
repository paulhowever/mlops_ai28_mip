from dataclasses import dataclass, field

from dota_winprob.agent.grounding import Evidence
from dota_winprob.agent.redaction import Redactor
from dota_winprob.agent.schemas import Step


@dataclass
class AgentDeps:
    redactor: Redactor
    evidence: Evidence = field(default_factory=Evidence)
    steps: list[Step] = field(default_factory=list)
