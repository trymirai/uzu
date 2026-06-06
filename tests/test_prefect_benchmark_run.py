from __future__ import annotations

import sys
import unittest
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from prefect_benchmark_run import PrefectApiClient, deployment_name_for_pool  # noqa: E402
from prefect_flow_run_common import PoolName  # noqa: E402


class RecordingPrefectApiClient(PrefectApiClient):
    def __init__(self) -> None:
        super().__init__(
            api_url="https://prefect.example/api",
            api_key="token",
            ui_base_url="https://prefect.example",
        )
        self.requests: list[dict[str, Any]] = []

    def _request_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        self.requests.append({"method": method, "path": path, "body": body})
        return {
            "id": "flow-run-id",
            "name": "flow-run-name",
            "state_type": "PENDING",
            "state_name": "Pending",
        }


class PrefectBenchmarkRunTests(unittest.TestCase):
    def test_new_max_pools_route_to_worker_deployments(self) -> None:
        self.assertIn("macos-m3-max-pool", {pool.value for pool in PoolName})
        self.assertIn("macos-m4-max-pool", {pool.value for pool in PoolName})
        self.assertEqual(
            deployment_name_for_pool(PoolName("macos-m3-max-pool")),
            "run-benchmark-worker-flow/run-benchmark-m3-max",
        )
        self.assertEqual(
            deployment_name_for_pool(PoolName("macos-m4-max-pool")),
            "run-benchmark-worker-flow/run-benchmark-m4-max",
        )

    def test_create_flow_run_sends_worker_flow_parameters(self) -> None:
        client = RecordingPrefectApiClient()

        client.create_flow_run_from_deployment(
            "deployment-id",
            (
                "run",
                "splitk-msmlx-m4-pro-pool",
                "--task",
                "london-summary",
                "--profile",
                "qwen35_uzu_vs_mlx_fast",
                "--uzu-branch",
                "gemm_perf_update",
            ),
        )

        self.assertEqual(
            client.requests,
            [
                {
                    "method": "POST",
                    "path": "/deployments/deployment-id/create_flow_run",
                    "body": {
                        "parameters": {
                            "identifier": "splitk-msmlx-m4-pro-pool",
                            "task": "london-summary",
                            "profile": "qwen35_uzu_vs_mlx_fast",
                            "uzu_ref": "gemm_perf_update",
                            "upload_metrics": False,
                        }
                    },
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
