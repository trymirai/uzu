from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from prefect_flow_run_common import PoolName  # noqa: E402
from prefect_uzu_cargo_run import deployment_name_for_pool  # noqa: E402


class PrefectUzuCargoRunTests(unittest.TestCase):
    def test_new_max_pools_route_to_worker_deployments(self) -> None:
        self.assertEqual(
            deployment_name_for_pool(PoolName.MACOS_M3_MAX),
            "run-uzu-cargo-command-worker-flow/run-uzu-cargo-command-m3-max",
        )
        self.assertEqual(
            deployment_name_for_pool(PoolName.MACOS_M4_MAX),
            "run-uzu-cargo-command-worker-flow/run-uzu-cargo-command-m4-max",
        )

    def test_workflow_exposes_new_max_pool_choices(self) -> None:
        workflow = (ROOT_DIR / ".github/workflows/run-uzu-cargo-command.yml").read_text()

        self.assertIn("          - macos-m3-max-pool\n", workflow)
        self.assertIn("          - macos-m4-max-pool\n", workflow)


if __name__ == "__main__":
    unittest.main()
