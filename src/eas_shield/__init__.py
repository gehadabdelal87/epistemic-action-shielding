"""Epistemic Action Shielding research prototype."""

from .actions import ActionLibrary, ActionSchema
from .events import EventModel, product_update
from .formulas import And, Atom, Formula, Knows, Not, Or, Top, conjunction, disjunction
from .governance import GovernancePolicy
from .model import EpistemicModel, PointedState
from .revision import EvidencePolicy, Observation
from .shield import DecisionMode, EASDecisionEngine
from .trace import DecisionTrace

__all__ = [
    "ActionLibrary",
    "ActionSchema",
    "And",
    "Atom",
    "DecisionMode",
    "DecisionTrace",
    "EASDecisionEngine",
    "EpistemicModel",
    "EvidencePolicy",
    "EventModel",
    "Formula",
    "GovernancePolicy",
    "Knows",
    "Not",
    "Observation",
    "Or",
    "PointedState",
    "Top",
    "conjunction",
    "disjunction",
    "product_update",
]

__version__ = "0.1.0"
