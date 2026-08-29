# Epistemic Action Shielding (EAS)

This repository is a research-grade reference implementation of **Epistemic Action Shielding for Autonomous Agents Under Uncertainty**. It implements the formal and experimental architecture developed in the manuscript:

- finite S5 epistemic models;
- cached model checking for agent-indexed knowledge;
- event-indexed product update;
- evidence admission and information-partition refinement;
- action-specific epistemic and environmental preconditions;
- machine-checkable operational-governance constraints;
- fail-closed ordinary-action and fallback selection;
- planner-independent external-policy shielding;
- authorization/execution separation;
- structured counterexample witnesses;
- deterministic trace replay;
- controlled diagnostic variants;
- stratified S5 scenario generation;
- threshold sensitivity and controlled scaling experiments; and
- an independently trained Deep Q-Network (DQN) policy-integration study.

The artifact is designed for formal inspection, controlled experiments, and reproducibility. It is **not** a deployment-ready safety system and does not solve the perception-to-epistemic mapping problem.

## Repository structure

```text
src/eas_shield/
  formulas.py             Epistemic formula language
  model.py                S5 models, validation, model checking, witnesses
  events.py               Dynamic epistemic event models and product update
  revision.py             Evidence admission and S5 partition refinement
  actions.py              Versioned action schemas and libraries
  governance.py           Operational-governance constraints
  policy.py               Proposal policies, legacy tabular Q-learning, and DQN
  shield.py               Fail-closed EAS authorization/execution engine
  trace.py                Authorization and execution trace records
  replay.py               Deterministic authorization replay
  environment.py          Gate-control domain and transitions
  scenario_generation.py  Logically valid stratified scenario generation
  variants.py             Controlled baselines and ablations
  metrics.py              Explicit-denominator metrics and output utilities
experiments/
  run_primary_diagnostic.py
  run_scaling.py
  run_sensitivity.py
  run_update_revision_experiment.py
  run_policy_integration.py
  statistical_analysis.py
  make_figures.py
  run_all.py
configs/
  quick.json
  journal.json
  governance_policy.json
tests/
  unit, property, negative, replay, and policy smoke tests
scripts/
  run_from_config.py
  reproduce.sh
```

## Installation

Python 3.11 or later is required.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e '.[experiments,test]'
```

The formal core uses only the Python standard library. Matplotlib, NumPy, and PyTorch are experiment dependencies; PyTorch is required only for the DQN study.

## Run the tests

```bash
pytest
```

The test suite checks:

- S5 frame validation and factivity;
- truth/knowledge separation;
- event-update preservation;
- evidence-refinement preservation;
- layered set inclusion;
- fail-closed action selection;
- canonical-condition violations in the controlled ablations;
- exact trace replay and corrupted-trace detection; and
- DQN training/checkpoint behavior and confidence/knowledge separation;
- legacy tabular Q-learning smoke behavior.

## Quick reproducibility run

```bash
bash scripts/reproduce.sh configs/quick.json results/quick
```

or, through the convenience wrapper:

```bash
python experiments/run_all.py --quick
```

This performs a small end-to-end run suitable for verifying installation. The versioned `configs/quick.json` file is the single source of experiment parameters for the quick workflow.

## Full journal protocol

```bash
bash scripts/reproduce.sh configs/journal.json results/journal
```

or, through the convenience wrapper:

```bash
python experiments/run_all.py
```

For the archival journal run, use:

```bash
bash scripts/reproduce.sh configs/journal.json results/journal
```

The full protocol runs five independent primary seeds, threshold sensitivity experiments with scenario-level monotonicity verification, controlled scaling probes, the informational update/revision experiment, the DQN learned-policy integration experiment, paired statistical analysis, and figure generation. `configs/journal.json` is the single source of journal experiment parameters.

A successful `scripts/reproduce.sh` run also writes:

- `config_used.json`, the exact configuration consumed by the runner;

- `python_version.txt`, the Python interpreter version;

- `environment_freeze.txt`, the realized Python environment;

- `environment.json`, structured runtime-environment metadata; and

- `artifact_manifest.json`, containing SHA-256 checksums and byte sizes for the generated artifact files.

The final numerical values should be generated from the actual version-controlled submission checkout on the declared machine. Record the exact Git commit or release tag separately and associate it with the archived result artifact; do not invent revision identifiers.

## Primary diagnostic experiment

```bash
python experiments/run_primary_diagnostic.py \
  --seeds 2026,2027,2028,2029,2030 \
  --scenarios-per-seed 1000 \
  --output results/primary
```

The six paired conditions are:

- `eas`;
- `state_truth`;
- `confidence`;
- `operator_knowledge`;
- `no_gate`; and
- `no_governance`.

Every condition receives the same scenario state, utility map, fallback ordering, and random seed. Each diagnostic variant modifies only the component named by the condition.

Outputs include:

- `decision_records.csv`;
- `aggregate_metrics.json`;
- `authorization_traces.json`;
- `scenario_strata.csv`; and
- `manifest.json`.

## Diagnostic separation of feasibility and safety

The gate domain treats `gate_operational` as environmental feasibility and `safe` as the proposition targeted by the epistemic requirement. This avoids making the state-truth substitution and no-gate ablation identical: a gate can be physically openable while unsafe, and EAS blocks opening unless the required agent-indexed knowledge condition holds.

## Metric interpretation

The experiment deliberately avoids treating agreement with EAS as independent “accuracy.” It reports:

- epistemic violation rate;
- environmental violation rate;
- governance violation rate;
- full admissibility compliance;
- open and unsafe-open rates;
- fallback and halt burden;
- task completion;
- constrained regret;
- agent-substitution violations;
- proposal intervention and admissible-proposal override;
- exact replay success; and
- authorization runtime.

Every rate is accompanied by its numerator and denominator in the aggregate output. If a denominator is zero, the JSON value is `null` rather than a non-standard `NaN`, while the numerator and denominator remain available for interpretation.

## Scaling

```bash
python experiments/run_scaling.py --repetitions 100 --output results/scaling
```

The scaling script varies one factor at a time:

- number of worlds;
- number of agents;
- number of contextual propositions; and
- number of actions.

Model generation is excluded from authorization timing. The script reports median, Q1, Q3, IQR, mean, and p95 authorization time, together with peak traced memory, relation-edge count, formula size, and trace size.

## Sensitivity

```bash
python experiments/run_sensitivity.py --scenarios 1000 --output results/sensitivity
```

The same scenario parameter set is reused while varying one threshold at a time:

- source reliability;
- observation quality; and
- maximum admissible action risk.

The script verifies monotonicity scenario by scenario rather than inferring it from aggregate rates. It writes `sensitivity_monotonicity.json`, records verification status in the sensitivity manifest, and fails the run if a required subset relation is violated. This also exposes the tradeoff among ordinary action, fallback, task completion, and constrained regret.

## Informational update and revision experiment

```bash
python experiments/run_update_revision_experiment.py \
  --repetitions 500 \
  --output results/update_revision
```

This experiment separates no information, public announcement, strong robot observation, weak robot observation, and strong operator-only observation. It verifies that action availability changes only when the relevant agent's represented epistemic state changes.

## DQN learned-policy integration

```bash
python experiments/run_policy_integration.py \
  --training-episodes 5000 \
  --evaluation-episodes 100 \
  --training-seeds 2026,2027,2028,2029,2030,2031,2032,2033,2034,2035 \
  --output results/policy
```

A DQN is trained in the partially observable three-action gate environment without access to EAS. The policy is frozen before evaluation and the same policy is then tested under three proposal-to-execution conditions:

- `unshielded`: the DQN proposal is executed directly;
- `confidence`: `open` is blocked only when scalar safety confidence falls below the configured threshold; and
- `eas`: the DQN proposal is passed through EAS, where `open` requires the robot to know that the gate is safe.

The three actions are `open`, `inspect`, and `defer`. Initial sensor evidence may be highly confident but does not by itself refine the robot's possible-world relation. `inspect` can return a truthful verification certificate or an inconclusive result; only verified evidence changes the epistemic model.

Evaluation includes an in-distribution sensor condition, a degraded but calibrated sensor, and a degraded overconfident sensor. Conditions are matched within each trained policy, while independent training seeds receive distinct evaluation episode seeds. The outputs report mean return, safe completion, unsafe opening, epistemically unsupported opening, epistemic luck, intervention, inspection, deferral, DQN inference time, and EAS authorization time.

The DQN study is intended to test runtime integration and the separation between learned preference, scalar confidence, and epistemic admissibility. It is not a claim of state-of-the-art reinforcement-learning performance.

## Paired statistical analysis

```bash
python experiments/statistical_analysis.py \
  --records results/journal/primary/decision_records.csv \
  --output results/journal/statistics
```

Binary paired outcomes use exact McNemar tests. Continuous paired outcomes use bootstrap confidence intervals. Holm-adjusted values are included for the binary comparison family.

## Figures

```bash
python experiments/make_figures.py \
  --results results/journal \
  --output results/journal/figures \
  --format png
```

All figures are regenerated from archived machine-readable results. No manuscript figure should be edited manually after generation.

## Trace replay

Every authorization trace contains a replay bundle with:

- the pointed authorization state;
- action-library version;
- governance-policy version;
- proposal and mode;
- utility map;
- fallback ordering; and
- random seed.

```python
from eas_shield.replay import replay_authorization

result = replay_authorization(trace_entry)
assert result.success
```

Replay establishes procedural reconstructibility relative to the stored model and specification. It does not establish that the model accurately represents the physical world.

## Important guarantee boundary

The implementation establishes this conditional property:

> If the model, action schemas, operational constraints, and software implementation are correct, every executed non-halt action satisfies the represented epistemic, environmental, and governance conditions attached to it.

It does not establish that:

- an S5 model is the correct representation for every agent;
- sensor or learned representations have been mapped correctly into propositions and information cells;
- admitted evidence is true in the external world;
- the governance predicates are legally or normatively complete;
- the transition model is accurate;
- explicit possible-world enumeration scales to unrestricted deployment; or
- formal compliance alone makes a system practically or ethically adequate.

## S5 and misleading observations

The main formalization uses factive S5 knowledge. It therefore cannot represent a false proposition as known at the designated world. Misleading observations may affect confidence, evidence admission, or an alternative belief-model extension, but they do not become false S5 knowledge. A deployment requiring false belief should add a KD45 or probabilistic-belief layer rather than mislabeling false belief as knowledge.

## Archival release

Before journal submission:

1. create the public/version-controlled repository and commit the exact artifact to be evaluated;
2. run `bash scripts/reproduce.sh configs/journal.json results/journal` from that checkout so the configuration snapshot, runtime environment, and output checksums are captured;
3. verify that `results/journal/` contains `config_used.json`, `python_version.txt`, `environment_freeze.txt`, `environment.json`, `artifact_manifest.json`, and the `primary/`, `scaling/`, `sensitivity/`, `update_revision/`, `policy/`, `statistics/`, and `figures/` directories;
4. regenerate manuscript tables and figures only from the archived machine-readable outputs;
5. create a frozen public release and archive it with a DOI or other persistent identifier once one is actually minted; and
6. include the real release identifier and Git commit in the manuscript’s code-availability statement.

Do not invent or manually backfill a DOI, repository URL, or Git revision before those identifiers exist, and do not hand-edit `artifact_manifest.json` after generation.

## License

MIT License. See `LICENSE`.
