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
from mlrun.datastore.datastore_profile import DatastoreProfileS3
from tests.system.base import TestMLRunSystem

test_environment = TestMLRunSystem._get_env_from_file()


@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.skipif(
    not test_environment.get("AWS_ACCESS_KEY_ID"),
    reason="AWS_ACCESS_KEY_ID is not set",
)
@pytest.mark.skipif(
    not test_environment.get("AWS_SECRET_ACCESS_KEY"),
    reason="AWS_SECRET_ACCESS_KEY is not set",
)
@pytest.mark.skipif(
    not test_environment.get("AWS_BUCKET_NAME"),
    reason="AWS_BUCKET_NAME is not set",
)
class TestCodeArtifact(TestMLRunSystem):
    """System tests for loading functions from store:// code artifacts (ML-11980)."""

    project_name = "code-artifact-system-test"
    image = "mlrun/mlrun"

    def _s3_path(self, filename: str) -> str:
        bucket = test_environment["AWS_BUCKET_NAME"]
        run_id = uuid.uuid4().hex[:8]
        return f"s3://{bucket}/test-code-artifact/{run_id}/{filename}"

    def _upload_code(self, s3_path: str, code: str):
        mlrun.get_dataitem(s3_path).put(code.encode())

    def test_job_function_from_store_artifact_s3(self):
        """Job function loads code from store:// artifact pointing to S3."""
        code = (
            "def handler(context):\n"
            "    context.logger.info('hello from artifact')\n"
            "    context.log_result('return', 42)\n"
        )
        s3_path = self._s3_path("job_func.py")
        self._upload_code(s3_path, code)

        self.project.log_code_file(
            "job-func-code",
            target_path=s3_path,
            language="python",
            code_type="function",
        )

        func = self.project.set_function(
            func=f"store://artifacts/{self.project_name}/job-func-code",
            name="job-from-artifact",
            kind="job",
            handler="handler",
            image=self.image,
        )

        run = func.run(local=False)
        assert run.status.state == "completed"
        assert run.status.results.get("return") == 42

        # Verify store:// is preserved in DB
        db_func = self.project.get_function("job-from-artifact")
        assert mlrun.datastore.is_store_uri(db_func.spec.build.source)

    def test_job_function_from_store_artifact_ds_profile(self):
        """Job function loads code via ds:// datastore profile."""
        bucket = test_environment["AWS_BUCKET_NAME"]
        profile = DatastoreProfileS3(
            name="test-code-profile",
            bucket=bucket,
            access_key_id=test_environment["AWS_ACCESS_KEY_ID"],
            secret_key=test_environment["AWS_SECRET_ACCESS_KEY"],
        )
        self.project.register_datastore_profile(profile)

        run_id = uuid.uuid4().hex[:8]
        ds_path = f"ds://test-code-profile/test-code-artifact/{run_id}/ds_func.py"

        code = (
            "def handler(context):\n"
            "    context.log_result('return', 'ds-profile-works')\n"
        )
        mlrun.get_dataitem(ds_path).put(code.encode())

        self.project.log_code_file(
            "ds-func-code",
            target_path=ds_path,
            language="python",
            code_type="function",
        )

        func = self.project.set_function(
            func=f"store://artifacts/{self.project_name}/ds-func-code",
            name="job-from-ds-profile",
            kind="job",
            handler="handler",
            image=self.image,
        )

        run = func.run(local=False)
        assert run.status.state == "completed"
        assert run.status.results.get("return") == "ds-profile-works"

    def test_nuclio_function_from_store_artifact(self):
        """Nuclio function uses init container to load code from store:// artifact."""
        code = (
            "def handler(context, event):\n"
            "    return context.Response(\n"
            "        body='from-artifact',\n"
            "        content_type='text/plain',\n"
            "    )\n"
        )
        s3_path = self._s3_path("nuclio_func.py")
        self._upload_code(s3_path, code)

        self.project.log_code_file(
            "nuclio-code",
            target_path=s3_path,
            language="python",
            code_type="function",
        )

        func = self.project.set_function(
            func=f"store://artifacts/{self.project_name}/nuclio-code",
            name="nuclio-from-artifact",
            kind="nuclio",
            handler="handler:handler",
            image=self.image,
        )
        self.project.deploy_function("nuclio-from-artifact")

        resp = func.invoke("")
        assert resp.decode() == "from-artifact"

        # Verify no credentials leaked in function spec
        db_func = self.project.get_function("nuclio-from-artifact")
        for key, value in (db_func.spec.config or {}).items():
            assert "AWS_SECRET" not in str(value), f"Credential leak in {key}"

    def test_shared_artifact_across_functions(self):
        """Two job functions reference the same store:// artifact."""
        code = (
            "def handler(context):\n"
            "    context.log_result('return', 'shared-code')\n"
        )
        s3_path = self._s3_path("shared.py")
        self._upload_code(s3_path, code)

        self.project.log_code_file(
            "shared-code",
            target_path=s3_path,
            language="python",
            code_type="function",
        )

        store_uri = f"store://artifacts/{self.project_name}/shared-code"

        func_a = self.project.set_function(
            func=store_uri, name="func-a", kind="job",
            handler="handler", image=self.image,
        )
        func_b = self.project.set_function(
            func=store_uri, name="func-b", kind="job",
            handler="handler", image=self.image,
        )

        run_a = func_a.run(local=False)
        run_b = func_b.run(local=False)

        assert run_a.status.state == "completed"
        assert run_b.status.state == "completed"
        assert run_a.status.results.get("return") == "shared-code"
        assert run_b.status.results.get("return") == "shared-code"

    def test_redeploy_picks_up_updated_artifact(self):
        """After updating artifact code at the same S3 path, next run() gets the new version."""
        s3_path = self._s3_path("versioned.py")

        # V1
        self._upload_code(
            s3_path, "def handler(context):\n    context.log_result('return', 'v1')\n"
        )
        self.project.log_code_file(
            "versioned-code", target_path=s3_path, language="python",
        )

        func = self.project.set_function(
            func=f"store://artifacts/{self.project_name}/versioned-code",
            name="versioned-func",
            kind="job",
            handler="handler",
            image=self.image,
        )
        run1 = func.run(local=False)
        assert run1.status.results.get("return") == "v1"

        # V2 -- update the file at the same S3 path
        self._upload_code(
            s3_path, "def handler(context):\n    context.log_result('return', 'v2')\n"
        )

        # Next run should get V2 (job downloads fresh each time)
        run2 = func.run(local=False)
        assert run2.status.results.get("return") == "v2"
