from types import SimpleNamespace

from eligsir.contracts.control import ModalityEligibility
from eligsir.contracts.runtime import SourceIdentity
from eligsir.runtime.production_application import CanonicalMappingRuntime


def test_post_refinement_catalog_rebuilds_only_when_db_generation_changes():
    """Refinement rearms must not rescan unchanged durable history."""

    modality = ModalityEligibility(rgb_supervision=True, rgb_quality="qualified")
    sources = tuple(
        SourceIdentity("camera", 0, ordinal, ordinal)
        for ordinal in range(10_000)
    )

    class Database:
        historical_rgb_materializable_generation = 1
        calls = 0

        def historical_candidates(self, **_kwargs):
            self.calls += 1
            return tuple(SimpleNamespace(frame_id=ordinal) for ordinal in range(10_000))

    database = Database()
    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = database
    runtime._historical_sources_by_frame_id = dict(enumerate(sources))
    runtime._replay_modalities = {source: modality for source in sources}

    for _ in range(1_000):
        assert len(runtime._post_refinement_replay_catalog()) == 10_000

    assert database.calls == 1

    database.historical_rgb_materializable_generation = 2
    assert len(runtime._post_refinement_replay_catalog()) == 10_000
    assert database.calls == 2


def test_prepared_post_refinement_catalog_excludes_unpageable_scene_history():
    """A frozen READY plan may name only COLD records the prepared path can rebuild."""

    modality = ModalityEligibility(rgb_supervision=True, rgb_quality="qualified")
    pagein_ready = SourceIdentity("camera", 0, 1, 1)
    scene_only = SourceIdentity("camera", 0, 2, 2)

    class Database:
        historical_rgb_materializable_generation = 1

        def historical_candidates(self, **_kwargs):
            return (
                SimpleNamespace(frame_id=1),
                SimpleNamespace(frame_id=2),
            )

        @staticmethod
        def get_frame(frame_id):
            if frame_id == 1:
                return SimpleNamespace(
                    semantic_action="defer",
                    prepared_defer_context=object(),
                    prepared_commit_context=None,
                    prepared_commit_obligation=None,
                )
            return SimpleNamespace(
                semantic_action=None,
                prepared_defer_context=None,
                prepared_commit_context=None,
                prepared_commit_obligation=None,
            )

    runtime = object.__new__(CanonicalMappingRuntime)
    runtime._observation_db = Database()
    runtime._historical_sources_by_frame_id = {
        1: pagein_ready,
        2: scene_only,
    }
    runtime._replay_modalities = {
        pagein_ready: modality,
        scene_only: modality,
    }
    runtime._ready_source_owner = object()

    assert tuple(runtime._post_refinement_replay_catalog()) == (pagein_ready,)
