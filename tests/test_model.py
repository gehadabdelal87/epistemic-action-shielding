from eas_shield.formulas import Atom, Knows, Not, conjunction
from eas_shield.model import EpistemicModel, ModelValidationError, equivalence_relation_from_partition


def simple_model():
    worlds = ("safe", "unsafe")
    relation = equivalence_relation_from_partition(worlds, (worlds,))
    model = EpistemicModel(
        worlds=worlds,
        agents=("robot",),
        propositions=("p",),
        valuation={"safe": frozenset({"p"}), "unsafe": frozenset()},
        relations={"robot": relation},
    )
    model.validate()
    return model


def test_truth_does_not_imply_knowledge():
    model = simple_model()
    checker = model.checker()
    assert checker.satisfies("safe", Atom("p"))
    assert not checker.satisfies("safe", Knows("robot", Atom("p")))
    witness = checker.counterexample("safe", Knows("robot", Atom("p")))
    assert witness is not None
    assert witness.world == "unsafe"


def test_factivity_in_s5():
    worlds = ("w",)
    model = EpistemicModel(
        worlds=worlds,
        agents=("robot",),
        propositions=("p",),
        valuation={"w": frozenset({"p"})},
        relations={"robot": {"w": frozenset({"w"})}},
    )
    model.validate()
    checker = model.checker()
    assert checker.satisfies("w", Knows("robot", Atom("p")))
    assert checker.satisfies("w", Atom("p"))


def test_conservative_accessibility_overapproximation_preserves_admitted_knowledge():
    """If knowledge survives a larger possibility set, it survives refinement.

    This is the finite-model sanity check behind the conservative-abstraction
    argument used in the manuscript: the approximating relation contains all
    worlds in the true information cell and may contain additional worlds.
    """

    worlds = ("w0", "w1", "w2")
    valuation = {
        "w0": frozenset({"p"}),
        "w1": frozenset({"p"}),
        "w2": frozenset(),
    }
    approximate = EpistemicModel(
        worlds=worlds,
        agents=("robot",),
        propositions=("p",),
        valuation=valuation,
        relations={
            "robot": equivalence_relation_from_partition(
                worlds, (("w0", "w1"), ("w2",))
            )
        },
    )
    refined = EpistemicModel(
        worlds=worlds,
        agents=("robot",),
        propositions=("p",),
        valuation=valuation,
        relations={
            "robot": equivalence_relation_from_partition(
                worlds, (("w0",), ("w1",), ("w2",))
            )
        },
    )
    approximate.validate()
    refined.validate()

    assert approximate.satisfies("w0", Knows("robot", Atom("p")))
    assert refined.satisfies("w0", Knows("robot", Atom("p")))


def test_rejects_non_reflexive_relation():
    model = EpistemicModel(
        worlds=("w",),
        agents=("robot",),
        propositions=("p",),
        valuation={"w": frozenset()},
        relations={"robot": {"w": frozenset()}},
    )
    try:
        model.validate()
    except ModelValidationError:
        pass
    else:
        raise AssertionError("Expected ModelValidationError")


def test_formula_roundtrip_in_model_serialization():
    model = simple_model()
    restored = EpistemicModel.from_dict(model.to_dict())
    assert restored.model_id == model.model_id
    assert restored.satisfies(
        "safe", conjunction(Atom("p"), Not(Knows("robot", Not(Atom("p")))))
    )
