from types import SimpleNamespace

from eligsir.map.arbiter import (
    DelegatedMutationEffect,
    DelegatedMutationProposal,
)
from eligsir.utils.strategies.mcmc.arbiter_owned import ArbiterOwnedMCMCStrategy


def test_builder_emits_one_delegated_proposal_and_reports_exact_effect():
    calls = []

    class _Strategy:
        name = "test-mcmc"

        def step_post_backward(self, **kwargs):
            calls.append(kwargs)
            return {"relocated": 3, "added": 5, "content_changed": True}

    wrapper = object.__new__(ArbiterOwnedMCMCStrategy)
    wrapper._strategy = _Strategy()
    wrapper._backend = object()
    wrapper._args = SimpleNamespace(cap_max=100)
    wrapper.name = "test-mcmc"

    proposal = wrapper.delegated_post_backward_proposal(iteration=17)
    effect = proposal.executor()

    assert isinstance(proposal, DelegatedMutationProposal)
    assert calls[0]["map_backend"] is wrapper._backend
    assert effect == DelegatedMutationEffect(
        content_changed=True,
        topology_changed=True,
        relocated=3,
    )


def test_existing_step_post_backward_commits_through_builder(monkeypatch):
    wrapper = object.__new__(ArbiterOwnedMCMCStrategy)
    wrapper._backend = SimpleNamespace()
    wrapper._args = SimpleNamespace(cap_max=100)
    wrapper.name = "test-mcmc"
    proposal = DelegatedMutationProposal(
        key=SimpleNamespace(),
        executor=lambda: "done",
    )
    wrapper.delegated_post_backward_proposal = lambda **_kwargs: proposal
    commits = []

    class _Arbiter:
        def __init__(self, backend, *, max_population):
            assert backend is wrapper._backend
            assert max_population == 100

        def commit(self, proposals):
            commits.append(proposals)

    monkeypatch.setattr(
        "utils.strategies.mcmc.arbiter_owned.GaussianMutationArbiter", _Arbiter
    )

    wrapper.step_post_backward(iteration=17)

    assert commits == [(proposal,)]
