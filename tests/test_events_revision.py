from eas_shield.events import public_announcement_event_model, product_update
from eas_shield.formulas import Atom, Knows
from eas_shield.model import EpistemicModel, PointedState, equivalence_relation_from_partition
from eas_shield.revision import EvidencePolicy, Observation, refine_information_partition


def state(designated_world="safe"):
    worlds = ("safe", "unsafe")
    relation = equivalence_relation_from_partition(worlds, (worlds,))
    model = EpistemicModel(
        worlds=worlds,
        agents=("robot",),
        propositions=("safe",),
        valuation={"safe": frozenset({"safe"}), "unsafe": frozenset()},
        relations={"robot": relation},
    )
    model.validate()
    return PointedState(model, designated_world, {})


def test_public_announcement_product_update():
    original = state()
    event = public_announcement_event_model(("robot",), Atom("safe"))
    updated = product_update(original, event, "announce")
    assert len(updated.model.worlds) == 1
    assert updated.model.satisfies(updated.world, Knows("robot", Atom("safe")))


def test_observation_refinement_preserves_s5_and_induces_knowledge():
    original = state()
    observation = Observation(
        observation_id="o1",
        recipient="robot",
        formula=Atom("safe"),
        observed_value=True,
        confidence=0.9,
        source="sensor",
        source_reliability=0.9,
        quality=0.9,
        provenance_known=True,
    )
    revised, record = refine_information_partition(original, observation, EvidencePolicy())
    revised.model.validate()
    assert record.admitted
    assert revised.model.satisfies(revised.world, Knows("robot", Atom("safe")))


def test_weak_observation_does_not_revise():
    original = state()
    observation = Observation(
        observation_id="o2",
        recipient="robot",
        formula=Atom("safe"),
        observed_value=True,
        confidence=0.1,
        source="sensor",
        source_reliability=0.9,
        quality=0.9,
        provenance_known=True,
    )
    revised, record = refine_information_partition(original, observation, EvidencePolicy())
    assert not record.admitted
    assert revised.model.model_id == original.model.model_id


def test_high_confidence_misleading_observation_cannot_create_false_knowledge():
    original = state("unsafe")
    observation = Observation(
        observation_id="misleading-high-confidence",
        recipient="robot",
        formula=Atom("safe"),
        observed_value=True,
        confidence=0.99,
        source="sensor",
        source_reliability=0.99,
        quality=0.99,
        provenance_known=True,
    )
    revised, record = refine_information_partition(original, observation, EvidencePolicy())
    revised.model.validate()

    assert record.admitted
    assert not revised.model.satisfies(
        revised.world, Knows("robot", Atom("safe"))
    )
