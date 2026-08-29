# EAS Architecture Notes

## Authorization pipeline

The implementation follows this sequence:

```text
validate
  -> event update
  -> evidence revision
  -> epistemic gate
  -> environmental filter
  -> operational-governance filter
  -> ordinary-action or fallback selection
  -> authorization trace
  -> action execution
  -> execution trace
```

Authorization and execution are deliberately separate. A favorable outcome cannot retroactively make an inadmissible authorization admissible, and an execution failure does not by itself prove that the authorization rule was incorrect.

## Model assumptions

The main model class is finite S5. Every agent relation must be an equivalence relation. This gives factivity and prevents vacuous knowledge caused by empty accessibility sets.

Alternative belief semantics should be implemented as a separate model class rather than weakening the meaning of the existing `Knows` operator.

## Event update

`events.py` implements product update. Postconditions are formulas evaluated at the pre-update world. Unspecified proposition postconditions default to identity.

## Revision

The prototype evidence revision intersects one agent's information partition with agreement on the admitted observation content. This preserves equivalence. Evidence admission depends on confidence, source reliability, observation quality, and provenance.

Admission is not a proof that the evidence is externally true. The simulator oracle is separated from runtime policy input.

## Governance

The governance layer implements selected machine-checkable operational predicates. Each result includes a stable identifier, Boolean outcome, structured reason, evidence, and version.

## Fallback

Fallback actions are ordinary schemas with their own preconditions and constraints. The algorithm returns `halt` if no ordinary or fallback action is admissible.

## Replay

Authorization traces include a complete replay bundle. Replay reconstructs the authorization state, action library, governance policy, mode, proposal, utilities, and fallback ordering, then recomputes all action sets and the selected response.
