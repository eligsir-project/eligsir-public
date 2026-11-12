"""Static Docker/Viser packaging and publication contracts."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_viser_build_option_installs_and_verifies_package():
    dockerfile = (ROOT / "Dockerfile").read_text()
    tail = dockerfile[dockerfile.index("ARG INSTALL_VISER"):]
    start = tail.index('if [ "${INSTALL_VISER}" = "true" ]')
    block = tail[start:start + 400]
    for token in ("pip install", '"viser', "import viser"):
        assert token in block, f"Missing Viser build contract: {token}"


def test_compose_exposes_build_option_and_independent_configurable_ports():
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "INSTALL_VISER" in compose
    assert '${VISER_HOST_PORT:-8080}:${VISER_CONTAINER_PORT:-8080}' in compose
    assert '"8080:8080"' not in compose
    assert "--viewer_viser_port" in compose
    assert "VISER_CONTAINER_PORT" in compose
