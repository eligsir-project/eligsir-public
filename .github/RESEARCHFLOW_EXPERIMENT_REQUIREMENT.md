# Mandatory ResearchFlow requirement for EliGSiR scientific experiments

All new EliGSiR experiments intended for scientific interpretation must be ResearchFlow-managed.

This includes:

- ablations;
- baseline/champion comparisons;
- benchmark runs;
- experiments supporting GitHub research issues;
- runs intended for paper figures or tables;
- automated scientific decisions and follow-up activation.

Direct EliGSiR, shell-script, or Ablator-only execution remains permitted for development, debugging, profiling, and smoke testing. Such runs are non-canonical unless they later pass the explicit ResearchFlow import/audit workflow.

Canonical scientific evidence must have a valid ResearchFlow identity before dispatch and must be ingested into the ResearchFlow Experiment Registry after canonical EliGSiR evidence and execution provenance are validated.

See `docs/researchflow_experiment_policy.md`.
