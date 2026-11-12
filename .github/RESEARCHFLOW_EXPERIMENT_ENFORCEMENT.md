# ResearchFlow enforcement checklist

The mandatory ResearchFlow scientific-evidence policy is **active**. This file is a regression checklist for the surfaces that must continue enforcing it; it is not a future activation gate.

Experiment-facing paths that can produce canonical scientific evidence must preserve all of the following:

- scientific experiment issue/template paths emit or require a ResearchFlow specification;
- managed experiment identity exists before dispatch;
- EliGSiR result/completion evidence is machine-readable and fail-closed;
- canonical comparison/baseline/champion/paper-result tooling accepts only ResearchFlow-managed or explicitly audited imported runs;
- unmanaged/debug runs remain usable but visibly non-canonical;
- no helper script or legacy workflow can bypass scientific eligibility by pointing directly at an arbitrary output directory;
- CI contains regression coverage for mandatory scientific-mode enforcement;
- paper/table/benchmark exports consume the same eligibility and comparability contract rather than maintaining a second result authority.

The durable policy is [`docs/researchflow_experiment_policy.md`](../docs/researchflow_experiment_policy.md).
