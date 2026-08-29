# Journal Experiment Protocol

## Primary experiment

Use at least five independent seeds. Generate models before deriving epistemic labels. Every diagnostic condition must receive the same scenario and utility configuration.

Conditions:

1. full EAS;
2. state-truth substitution;
3. confidence substitution;
4. operator-knowledge substitution;
5. no epistemic gate; and
6. no governance filter.

The canonical EAS action library is always used for evaluation metrics, including when a diagnostic condition changes its runtime action precondition.

## Required primary outcomes

- epistemic violation rate;
- environmental violation rate;
- governance violation rate;
- full admissibility compliance;
- task-completion rate;
- fallback and halt burden;
- utility and constrained regret;
- agent-substitution violation rate;
- runtime; and
- replay success.

Every rate must include a numerator and denominator.

## Scaling

Vary one factor at a time. Generate models outside the timed authorization block. Use warm-up runs. Report median, IQR, mean, p95, peak memory, relation edges, formula size, and trace size.

## Sensitivity

Reuse the same scenario parameter set while changing one threshold. Verify the expected monotonic subset relationship for stronger constraints.

## Policy integration

Train each DQN without EAS and freeze its parameters before evaluation. Use at least five independent training seeds; the journal configuration uses ten. For each trained policy, evaluate matched episode seeds under three execution conditions: unshielded DQN, scalar confidence-threshold gating, and EAS shielding. Episode seeds must match across the three conditions for a given policy but differ across independent training seeds.

Evaluate the policies under (i) the training sensor distribution, (ii) degraded but calibrated sensing, and (iii) degraded overconfident sensing. Report mean return, safe-completion rate, unsafe-action rate, epistemically unsupported-action rate, epistemic-luck rate, policy-intervention rate, inspection and deferral rates, DQN inference time, and EAS authorization overhead. Preserve trained checkpoints and per-step records so aggregate rates can be reconstructed.

## Archival requirements

- frozen source release;
- public DOI;
- exact command manifest;
- raw per-decision outputs;
- exact seeds;
- software and hardware details;
- generated figures; and
- trace replay script.
