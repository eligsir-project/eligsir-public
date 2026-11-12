# A clean one-commit publication snapshot

The working repository is **private** and has GitHub PR/workflow actor metadata.
Switching its visibility to public does not provide repository-level anonymity.
This tool deliberately **does not** fork, mirror, transfer or publish that repository.

## Make a scanned source archive and fresh local Git root

From a clean working tree:

~~~bash
python scripts/export_public_snapshot.py \
  --output /tmp/eligsir-source.tar.gz \
  --init-repo /tmp/eligsir-source-clean
python /tmp/eligsir-source-clean/scripts/audit_release_identity.py \
  --root /tmp/eligsir-source-clean --require-anonymous-history --all-refs
~~~

The export only includes files from the current committed tree. It checks the
archive bytes against high-confidence identity/credential patterns and rejects
symlinks and unsafe archive paths. The new local repository has exactly one
commit with a neutral name/email and no inherited refs or remotes. The archive
itself has **no** `.git` directory.

Before publishing, conduct a dedicated full secret scan, inspect embedded
binary metadata and review the intended code/license scope. The tool is a
conservative preflight, not an exhaustive guarantee.

## Uploading into a genuinely new remote

Create an **empty**, non-forked repository with the intended visibility using
an appropriate account. Do not initialize it with a README or license if
pushing the prepared snapshot, and do not use an author-identifying account or
remote if this artifact must satisfy a double-anonymous review.

In the clean local repository:

~~~bash
git remote add origin https://github.com/eligsir-project/eligsir-public.git
git push -u origin main
~~~

This step requires repository-creation permissions and is **not** performed by
the exporter. Public GitHub organization membership, commits attributed by the
host, Actions actors, issues, repository events, and linked project/preprint
pages can still disclose authorship, irrespective of local Git metadata.

Always obtain the applicable rights and scientific-release qualification before
turning on public visibility. See [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
