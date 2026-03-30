# Copyright 2023 Iguazio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import uuid

import pytest

import mlrun
import mlrun.datastore
from tests.system.base import TestMLRunSystem

test_environment = TestMLRunSystem._get_env_from_file()


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestCodeArtifact(TestMLRunSystem):
    """System tests for loading functions from store:// code artifacts (ML-11980)."""

    project_name = "code-artifact-system-test"
    # Use default mlrun/mlrun image — the cluster's installed mlrun image.
    # Don't specify an external registry path to avoid Nuclio runRegistry prefix issues.
    image = "mlrun/mlrun"

    @classmethod
    def custom_setup_class(cls):
        # Trigger migrations if needed (new deployment may require it)
        try:
            cls._run_db.trigger_migrations()
        except Exception:
            pass  # best effort — may already be migrated

    def _v3io_path(self, filename: str) -> str:
        run_id = uuid.uuid4().hex[:8]
        return f"v3io:///projects/{self.project_name}/test-code-artifact/{run_id}/{filename}"

    def _upload_code(self, path: str, code: str):
        mlrun.get_dataitem(path).put(code.encode())

    def _set_function(self, **kwargs):
        """Set function with the test image."""
        return self.project.set_function(image=self.image, **kwargs)

    def test_job_function_from_store_artifact(self):
        """Job function loads code from store:// artifact pointing to v3io."""
        code = (
            "def handler(context):\n"
            "    context.logger.info('hello from artifact')\n"
            "    context.log_result('return', 42)\n"
        )
        v3io_path = self._v3io_path("job_func.py")
        self._upload_code(v3io_path, code)

        self.project.log_code_file(
            "job-func-code",
            target_path=v3io_path,
            language="python",
            code_type="function",
        )

        func = self._set_function(
            func=f"store://artifacts/{self.project_name}/job-func-code",
            name="job-from-artifact",
            kind="job",
            handler="job_func.handler",
        )

        run = func.run(local=False, watch=False)
        run.wait_for_completion(timeout=300)
        if run.status.state == "error":
            self._logger.error(
                "Run failed",
                error=run.status.error,
                status=run.status.to_dict(),
            )
        assert run.status.state == "completed", f"Run failed: {run.status.error}"
        assert run.status.results.get("return") == 42

        # Verify store:// is preserved in DB
        db_func = self.project.get_function("job-from-artifact")
        assert mlrun.datastore.is_store_uri(db_func.spec.build.source)

    def test_nuclio_function_from_store_artifact(self):
        """Nuclio function with store:// — build-time mode (default).

        Server resolves artifact, embeds code as functionSourceCode.
        No init container needed.
        """
        code = (
            "def handler(context, event):\n"
            "    return context.Response(\n"
            "        body='from-artifact',\n"
            "        content_type='text/plain',\n"
            "    )\n"
        )
        v3io_path = self._v3io_path("nuclio_func.py")
        self._upload_code(v3io_path, code)

        self.project.log_code_file(
            "nuclio-code",
            target_path=v3io_path,
            language="python",
            code_type="function",
        )

        func = self._set_function(
            func=f"store://artifacts/{self.project_name}/nuclio-code",
            name="nuclio-from-artifact",
            kind="nuclio",
            handler="nuclio_func:handler",
        )
        # Default: load_source_on_run=False → build-time resolution
        self.project.deploy_function("nuclio-from-artifact")

        resp = func.invoke("")
        assert resp.decode() == "from-artifact"

        # Verify no credentials leaked in function spec
        db_func = self.project.get_function("nuclio-from-artifact")
        for key, value in (db_func.spec.config or {}).items():
            assert "V3IO_ACCESS_KEY" not in str(value), f"Credential leak in {key}"

    def test_nuclio_function_from_store_artifact_runtime_mode(self):
        """Nuclio function with store:// — runtime mode (load_source_on_run=True).

        Init container downloads code at pod startup. Wrapper imports handler
        from PYTHONPATH. Code updates without rebuild.
        """
        code = (
            "def handler(context, event):\n"
            "    return context.Response(\n"
            "        body='runtime-mode',\n"
            "        content_type='text/plain',\n"
            "    )\n"
        )
        v3io_path = self._v3io_path("nuclio_rt.py")
        self._upload_code(v3io_path, code)

        self.project.log_code_file(
            "nuclio-rt-code",
            target_path=v3io_path,
            language="python",
            code_type="function",
        )

        func = self._set_function(
            func=f"store://artifacts/{self.project_name}/nuclio-rt-code",
            name="nuclio-runtime-mode",
            kind="nuclio",
            handler="nuclio_rt:handler",
        )
        func.spec.build.load_source_on_run = True
        self.project.deploy_function("nuclio-runtime-mode")

        resp = func.invoke("")
        assert resp.decode() == "runtime-mode"

    def test_shared_artifact_across_functions(self):
        """Two job functions reference the same store:// artifact."""
        code = (
            "def handler(context):\n    context.log_result('return', 'shared-code')\n"
        )
        v3io_path = self._v3io_path("shared.py")
        self._upload_code(v3io_path, code)

        self.project.log_code_file(
            "shared-code",
            target_path=v3io_path,
            language="python",
            code_type="function",
        )

        store_uri = f"store://artifacts/{self.project_name}/shared-code"

        func_a = self._set_function(
            func=store_uri,
            name="func-a",
            kind="job",
            handler="shared.handler",
        )
        func_b = self._set_function(
            func=store_uri,
            name="func-b",
            kind="job",
            handler="shared.handler",
        )

        run_a = func_a.run(local=False, watch=False)
        run_b = func_b.run(local=False, watch=False)
        run_a.wait_for_completion(timeout=300)
        run_b.wait_for_completion(timeout=300)

        assert run_a.status.state == "completed", f"func-a failed: {run_a.status.error}"
        assert run_b.status.state == "completed", f"func-b failed: {run_b.status.error}"
        assert run_a.status.results.get("return") == "shared-code"
        assert run_b.status.results.get("return") == "shared-code"

    def test_redeploy_picks_up_updated_artifact(self):
        """After updating artifact code at the same path, next run() gets the new version."""
        v3io_path = self._v3io_path("versioned.py")

        # V1
        self._upload_code(
            v3io_path,
            "def handler(context):\n    context.log_result('return', 'v1')\n",
        )
        self.project.log_code_file(
            "versioned-code",
            target_path=v3io_path,
            language="python",
        )

        func = self._set_function(
            func=f"store://artifacts/{self.project_name}/versioned-code",
            name="versioned-func",
            kind="job",
            handler="versioned.handler",
        )
        run1 = func.run(local=False, watch=False)
        run1.wait_for_completion(timeout=300)
        assert run1.status.state == "completed", f"V1 run failed: {run1.status.error}"
        assert run1.status.results.get("return") == "v1"

        # V2 -- update the file at the same path
        self._upload_code(
            v3io_path,
            "def handler(context):\n    context.log_result('return', 'v2')\n",
        )

        # Next run should get V2 (job downloads fresh each time)
        run2 = func.run(local=False, watch=False)
        run2.wait_for_completion(timeout=300)
        assert run2.status.state == "completed", f"V2 run failed: {run2.status.error}"
        assert run2.status.results.get("return") == "v2"

    def test_job_function_from_store_artifact_with_requirements(self):
        """Job function from store:// artifact with requirements gets deps installed at build time."""
        code = (
            "import requests\n"
            "def handler(context):\n"
            "    context.log_result('requests_version', requests.__version__)\n"
        )
        v3io_path = self._v3io_path("func_with_deps.py")
        self._upload_code(v3io_path, code)

        self.project.log_code_file(
            "func-with-deps",
            target_path=v3io_path,
            language="python",
            code_type="function",
            requirements=["requests"],
        )

        func = self._set_function(
            func=f"store://artifacts/{self.project_name}/func-with-deps",
            name="job-with-deps",
            kind="job",
            handler="func_with_deps.handler",
        )

        run = func.run(local=False, watch=False)
        run.wait_for_completion(timeout=300)
        assert run.status.state == "completed", f"Run failed: {run.status.error}"
        assert run.status.results.get("requests_version")

    def test_nuclio_function_from_store_artifact_with_requirements(self):
        """Nuclio function from store:// artifact with requirements gets deps installed at build time."""
        code = (
            "import requests\n"
            "def handler(context, event):\n"
            "    return context.Response(\n"
            "        body=requests.__version__,\n"
            "        content_type='text/plain',\n"
            "    )\n"
        )
        v3io_path = self._v3io_path("nuclio_with_deps.py")
        self._upload_code(v3io_path, code)

        self.project.log_code_file(
            "nuclio-with-deps",
            target_path=v3io_path,
            language="python",
            code_type="function",
            requirements=["requests"],
        )

        func = self._set_function(
            func=f"store://artifacts/{self.project_name}/nuclio-with-deps",
            name="nuclio-with-deps",
            kind="nuclio",
            handler="nuclio_with_deps:handler",
        )
        self.project.deploy_function("nuclio-with-deps")

        resp = func.invoke("")
        # Should return the requests version string (not an import error)
        assert resp.decode().strip()
