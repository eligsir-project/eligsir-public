"""Keep executor wiring fakes at the explicit construction/finalization seam."""

from eligsir.mapping.integration.loss_diagnostics import GuardedLossAccumulator


def patch_assemble_loss(monkeypatch, objective, fake):
    def prepare(inputs):
        breakdown = fake(inputs)
        return GuardedLossAccumulator(), breakdown.total_tensor, lambda *_: breakdown

    # Some tests layer their fake over a previous fake; preserve that test-only API.
    monkeypatch.setattr(objective, "assemble_loss", fake)
    monkeypatch.setattr(objective, "prepare_loss", prepare)
