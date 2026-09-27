import pytest

from cascade.data.instances import ProcessedData
from cascade.data.preprocess import output_dir
from cascade.utils.config import REPO_ROOT, load_config


@pytest.fixture(scope="session")
def toy_data() -> ProcessedData:
    cfg = load_config(REPO_ROOT / "configs/data.yaml")
    if not (output_dir(cfg, True) / "instances.parquet").exists():
        pytest.skip("toy data not built; run scripts/prepare_data.sh --toy")
    return ProcessedData.load(str(REPO_ROOT / "configs/data.yaml"), toy=True)
