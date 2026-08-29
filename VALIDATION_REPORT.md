# Artifact Validation Report

Validation date: 2026-08-29

## Environment used for the current validation

The final validation and journal-scale reproduction were executed locally on the project machine in the active virtual environment.

- Python 3.13.0
- pytest 9.1.1
- NumPy 2.5.2
- Matplotlib 3.11.1
- PyTorch 2.13.0

The package metadata requires Python >=3.11. The exact environment used for the journal run is archived in `results/journal/environment_freeze.txt`, together with `environment.json` and `python_version.txt`.

`requirements-tested.txt` records the separately declared pinned validation set for the artifact; the journal reproduction itself preserves the exact packages actually installed on the execution machine.

## Validation completed

1. The repository was reconstructed into its intended package and experiment layout, and the package was installed in editable mode with:

   ```bash
   python -m pip install -e '.[experiments,test]'
   ```

2. The complete automated test suite completed with **25 passed, 0 failed** after the final code corrections.

3. A final quick end-to-end smoke reproduction completed successfully through `scripts/reproduce.sh`, exercising:

   - the primary diagnostic experiment;
   - controlled scaling;
   - threshold sensitivity with paired scenario-level monotonicity verification;
   - event-update and evidence-revision;
   - DQN learned-policy integration;
   - paired statistical analysis;
   - figure generation; and
   - provenance capture.

4. The final publication-scale reproduction completed successfully with:

   ```bash
   bash scripts/reproduce.sh configs/journal.json results/journal
   ```

5. The journal run produced the seven expected experiment/result groups:

   - `primary/`
   - `scaling/`
   - `sensitivity/`
   - `update_revision/`
   - `policy/`
   - `statistics/`
   - `figures/`

6. Figure generation produced **16 PNG figures**, including policy and return plots for all three DQN evaluation regimes:

   - in-distribution;
   - degraded calibrated; and
   - degraded overconfident.

7. The sensitivity experiment generated `sensitivity_monotonicity.json`. The journal-scale paired scenario schedule passed all adjacent admissible-set monotonicity checks with **0 violations** and `overall_passed: true`.

8. The scaling summary records runtime dispersion and structural diagnostics, including:

   - median authorization time;
   - Q1 and Q3 authorization time;
   - IQR;
   - mean authorization time;
   - p95 authorization time;
   - median and p95 peak memory;
   - relation edges;
   - formula count, size, and modal depth; and
   - median and p95 trace size.

9. All **13 JSON files** in `results/journal` were reparsed with a strict JSON parser that rejects non-standard constants such as `NaN` and `Infinity`. All 13 passed. Undefined aggregate rates are therefore archived as JSON `null`, with numerator and denominator fields retained.

10. The primary diagnostic uses the corrected `admissible_proposal_override` terminology, and the deprecated legacy metric label is absent from the active source, experiment, configuration, and publication-facing documentation.

11. The seeded policy path uses a deterministic SHA-256-derived seed offset rather than Python's randomized built-in `hash()`.

12. `scripts/reproduce.sh` generated the publication-scale provenance files:

   - `config_used.json`;
   - `python_version.txt`;
   - `environment_freeze.txt`;
   - `environment.json`; and
   - `artifact_manifest.json`.

13. The journal artifact manifest uses SHA-256 file hashing. A post-run verification recomputed every recorded digest: **50 files checked, all hashes matched**.

14. The journal result set contains the required publication artifacts, including:

   - `primary/aggregate_metrics.json`;
   - `scaling/scaling_summary.json`;
   - `sensitivity/sensitivity_monotonicity.json`;
   - `policy/dqn_summary.json`;
   - statistical paired-effect outputs;
   - raw primary, sensitivity, scaling, update/revision, and DQN records;
   - trained DQN checkpoints; and
   - all 16 figures.

15. The publication-scale result directory is intentionally excluded from normal Git tracking by `.gitignore`. The full `results/journal` artifact was archived outside the repository as:

   `EAS_journal_results_2026-08-29.tar.gz`

   The archive was successfully read with `tar -tzf` and passed the archive-integrity check.

16. The compressed journal archive is approximately **15 MB** (from approximately **472 MB** uncompressed). Its SHA-256 checksum is:

   `12a9a4ac548b59dde46952257a0f2b352a85c0694859bf860b24059746c30e01`

   The checksum is also stored beside the archive in:

   `EAS_journal_results_2026-08-29.tar.gz.sha256`

17. Generated research outputs remain excluded from version control by default. `results/smoke/README.md` is the intended tracked placeholder under `results/`.

## Journal-scale empirical checks

The journal-scale primary diagnostic produced the expected separation among layers:

- EAS: zero epistemic violations, zero governance violations, zero agent-substitution violations, and full admissibility compliance of 1.0.
- Removing the epistemic gate increased epistemic violations.
- Removing governance produced a substantial governance-violation rate.
- Replacing acting-agent knowledge with operator knowledge produced agent-substitution failures.
- Authorization remained sub-millisecond at the baseline primary-diagnostic scale.

The update/revision experiment also preserved agent-relative knowledge:

- no information: autonomous opening blocked;
- strong operator observation: operator knowledge established, robot knowledge absent, opening blocked;
- strong robot observation: robot knowledge established, opening licensed;
- public announcement: both agents know, opening licensed;
- weak robot observation: knowledge not established, opening blocked.

The DQN policy-integration experiment preserved the intended claim boundary:

- In-distribution, EAS eliminated unsafe and epistemically unsupported actions but imposed a return/completion tradeoff.
- Under degraded calibrated sensing, confidence gating and EAS produced the same reported safety outcomes.
- Under degraded overconfident sensing, confidence gating reverted to the unshielded behavior, whereas EAS continued to block unsafe and epistemically unsupported actions.

These results support the claim that EAS is **not** universally superior to a calibrated confidence gate. Its distinctive advantage appears when confidence ceases to track epistemic admissibility.

## Interpretation and scope

The quick run is a software integration and reproducibility check. Its numerical outputs are **not** manuscript results.

The manuscript should report the final `configs/journal.json` outputs and archive them together with the exact versioned repository state, configuration snapshot, environment records, raw outputs, statistical analyses, figures, trained policy checkpoints, and checksums.

The artifact is an explicit finite-model research prototype. Successful software validation does not establish deployment-scale scalability, real-world perceptual validity, or empirical superiority of EAS over a calibrated confidence gate in every regime.
