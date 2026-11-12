from pathlib import Path

from scene.readers.realityscan import is_realityscan_mobile


def test_is_realityscan_mobile_true_when_sfm_and_xmp_present(tmp_path):
    (tmp_path / "sfm.txt").write_text('RSv2,["fake"]')
    (tmp_path / "abc.xmp").write_text("<x/>")
    assert is_realityscan_mobile(str(tmp_path)) is True


def test_is_realityscan_mobile_false_without_sfm(tmp_path):
    (tmp_path / "abc.xmp").write_text("<x/>")
    assert is_realityscan_mobile(str(tmp_path)) is False


def test_is_realityscan_mobile_false_without_xmp(tmp_path):
    (tmp_path / "sfm.txt").write_text('RSv2,["fake"]')
    assert is_realityscan_mobile(str(tmp_path)) is False
