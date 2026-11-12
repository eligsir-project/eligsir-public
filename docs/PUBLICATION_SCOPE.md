# Publication documentation scope

This repository is a curated publication export, not a full source checkout.
The source project's large MkDocs corpus, including `docs/visuals/`, was
intentionally pruned. The remaining documentation is limited to release and
provenance records; it is not a complete generated documentation site.

`scripts/check_docs_integrity.py` recognizes this explicit marker and skips only
visual generation/audit; all other documentation ownership, API, and structure
checks remain active. A future documentation release must restore the visual
specifications and remove this marker before enabling visual checks.
