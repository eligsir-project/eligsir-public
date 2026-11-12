# EliGSiR publication-export guidance

This repository is a **curated release snapshot**, not the complete internal development checkout. The internal `AGENTS.md`, original Graft graph and extensive architecture documentation were intentionally excluded; do not follow broken links to them or infer that they ship with this export.

Use these retained sources:
- [README.md](README.md) — supported public interfaces, installation and status.
- [RELEASE_BLOCKERS.md](RELEASE_BLOCKERS.md) — unresolved software/rights/reproduction gates.
- [docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md) — publication and anonymity process.
- [docs/PROVENANCE_MATRIX.md](docs/PROVENANCE_MATRIX.md) — file-level license scope.
- [docs/THIRD_PARTY_PROVENANCE.yaml](docs/THIRD_PARTY_PROVENANCE.yaml) — dependency evidence and blockers.

Use executable code and tests as the source of truth for actual behavior. The separate `eligsir-ros2` project owns ROS 2 integration. Do not add real names, emails, host paths, unlicensed assets, or private service URLs to the release export.
