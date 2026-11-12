from eligsir.utils.checkpoint_io import unpack_checkpoint_payload


def test_unpack_2tuple_offline_checkpoint():
    model_params, iteration = unpack_checkpoint_payload(("fake-state", 30000))
    assert model_params == "fake-state"
    assert iteration == 30000


def test_unpack_3tuple_streaming_checkpoint_drops_meta():
    meta = {"n_frames_ingested": 5, "last_ingest_iter": 4950}
    model_params, iteration = unpack_checkpoint_payload(("fake-state", 200000, meta))
    assert model_params == "fake-state"
    assert iteration == 200000
