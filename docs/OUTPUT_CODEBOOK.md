# Output Codebook

## `primary/decision_records.csv`

- `scenario_id`: stable scenario identifier.
- `seed`: generation seed.
- `condition`: EAS or controlled diagnostic condition.
- `selected_action`: authorized action; empty for halt.
- `proposal`: external proposal when shield mode is used.
- `safe_world`: simulator safety truth for outcome evaluation only.
- `opened`: whether the selected action opens the gate.
- `fallback`: whether a fallback action was selected.
- `halted`: whether no action was authorized.
- `epistemic_violation`: selected action fails the canonical epistemic precondition.
- `environmental_violation`: selected action fails the canonical environmental precondition.
- `governance_violation`: selected action fails the canonical operational policy.
- `full_admissibility_compliance`: selected action passes all canonical layers.
- `agent_substitution_violation`: autonomous robot action uses operator knowledge while robot knowledge fails.
- `unsafe_open`: an opening action occurs in an unsafe simulator state.
- `task_completed`: domain goal criterion for the one-cycle diagnostic scenario.
- `utility`: configured nominal utility of the selected response.
- `constrained_regret`: regret relative to the best canonical admissible action; undefined for inadmissible selections.
- `proposal_intervened`: proposal differs from selected response.
- `proposal_inadmissible`: proposal is not canonically admissible.
- `inadmissible_proposal_blocked`: inadmissible proposal was not executed.
- `admissible_proposal_override`: admissible proposal was replaced or halted.
- `authorization_ns`: authorization runtime in nanoseconds.
- `trace_replay_success`: exact replay result.
- `status`: authorization status.

## `primary/aggregate_metrics.json`

Every rate includes fields with suffixes `_numerator` and `_denominator`.

## `primary/authorization_traces.json`

Contains replay bundles and hashes for each authorization decision.

## `scaling/scaling_records.csv`

Contains one timed authorization record per configuration and repetition. Scenario construction and file I/O are excluded from the timing block.

## `sensitivity/sensitivity_summary.csv`

Contains one aggregate row per parameter and threshold, using the same underlying scenario parameter schedule for all thresholds.

## `policy/dqn_step_records.csv`

Contains step-level records for the learned-policy experiment. Each row identifies the DQN training seed, evaluation regime, execution condition, hidden safety state, scalar confidence, inspection result, Q-values, proposed and executed actions, EAS intervention reason, epistemic/environmental/governance support, unsafe opening, epistemically unsupported opening, epistemic luck, reward, DQN inference time, and EAS authorization time.

## `policy/dqn_episode_records.csv`

Contains episode-level return, safe completion, harm, open count, unsupported and unsafe openings, epistemic luck, intervention count, inspection and deferral counts, and runtime summaries.

## `policy/dqn_training_records.csv`

Contains one row per independently trained DQN with the training seed, episode count, environment steps, gradient steps, final exploration rate, and final rolling return.

## `policy/dqn_summary.json`

Contains the complete experiment protocol and aggregate results for the unshielded, confidence-threshold, and EAS conditions under in-distribution, degraded-calibrated, and degraded-overconfident sensing. Aggregate metrics include bootstrap 95% confidence intervals across independent DQN training seeds.

## `statistics/paired_effects.csv`

Contains exact McNemar comparisons for binary outcomes and paired-bootstrap confidence intervals for continuous outcomes.
