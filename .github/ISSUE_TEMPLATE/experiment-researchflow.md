---
name: Scientific experiment (ResearchFlow)
about: Define a EliGSiR scientific experiment that must run through ResearchFlow
labels: research
---

# Experiment

## Scientific question / hypothesis

Describe the claim this experiment is intended to test.

## ResearchFlow requirement

This issue is a **scientific experiment**. Execution intended to contribute to comparisons, ablations, baselines/champions, benchmarks, paper figures/tables, or automated scientific decisions **must use ResearchFlow**.

Direct EliGSiR, shell-script, or Ablator-only runs are development/unmanaged evidence and are not canonical scientific results.

Add exactly one machine-readable `research` block below and use the ResearchFlow control-plane commands outside the block.

```research
version: 1
id: replace-me
type: experiment
automation: manual

requires: []
validity_gates: []

matrix: {}

run:
  profile: eligsir-streaming
  parameters: {}

evidence:
  telemetry: []
  metrics: []

acceptance: {}
followups: {}
```

/research plan

## Interpretation notes

Document expected positive/negative/inconclusive outcomes and any scientific caveats that should remain human-readable rather than machine semantics.

## Completion gate

- [ ] immutable ResearchFlow experiment/revision/plan/run/job identity exists before dispatch
- [ ] execution used the ResearchFlow → Snakemake → Ablator path (or an explicitly supported ResearchFlow execution backend)
- [ ] EliGSiR canonical completion/result evidence was ingested
- [ ] provenance/evidence/comparability gates were evaluated
- [ ] run is registered in the canonical ResearchFlow Experiment Registry
- [ ] deterministic aggregate/decision artifacts exist where applicable
- [ ] unmanaged/debug outputs were not used as canonical scientific evidence
