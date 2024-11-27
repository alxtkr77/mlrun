# Copyright 2024 Iguazio
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
#

import os
import tempfile

import pytest
import yaml

from mlrun.artifacts import DocumentLoaderSpec, MLRunLoader
from mlrun.datastore.datastore_profile import (
    VectorStoreProfile,
    register_temporary_client_datastore_profile,
)
from tests.system.base import TestMLRunSystem

here = os.path.dirname(__file__)
config_file_path = os.path.join(here, "../env.yml")

config = {}
if os.path.exists(config_file_path):
    with open(config_file_path) as yaml_file:
        config = yaml.safe_load(yaml_file)


@pytest.mark.skipif(
    not config.get("MILVUS_HOST") or not config.get("MILVUS_PORT"),
    reason="milvus parameters not configured",
)
# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestDatastoreProfile(TestMLRunSystem):
    def custom_setup(self):
        pass

    def test_vectorstore_document_artifact(self):
        # Create a temporary text file with a simple context
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write("This is a test document for vector store.")
            temp_file.flush()
            # Test logging a document localy
            artifact = self.project.log_document(
                "test_document_artifact", src_path=temp_file.name, upload=False
            )
            langchain_documents = artifact.to_langchain_documents()

            assert len(langchain_documents) == 1
            assert (
                langchain_documents[0].page_content
                == "This is a test document for vector store."
            )
            assert (
                langchain_documents[0].metadata["source"]
                == f"{self.project.name}/test_document_artifact"
            )
            assert langchain_documents[0].metadata["original_source"] == temp_file.name
            assert langchain_documents[0].metadata["mlrun_object_uri"] == artifact.uri
            assert langchain_documents[0].metadata["mlrun_chunk"] == "0"

            # Test logging a document localy
            artifact = self.project.log_document(
                "test_document_artifact", src_path=temp_file.name, upload=True
            )

            stored_artifcat = self.project.get_artifact("test_document_artifact")
            stored_langchain_documents = stored_artifcat.to_langchain_documents()

            assert (
                langchain_documents[0].page_content
                == stored_langchain_documents[0].page_content
            )
            assert (
                langchain_documents[0].metadata["source"]
                == stored_langchain_documents[0].metadata["source"]
            )
            assert (
                langchain_documents[0].metadata["original_source"]
                == stored_langchain_documents[0].metadata["original_source"]
            )
            assert (
                langchain_documents[0].metadata["mlrun_chunk"]
                == stored_langchain_documents[0].metadata["mlrun_chunk"]
            )
            assert (
                stored_langchain_documents[0].metadata["mlrun_object_uri"]
                == stored_artifcat.uri
            )
            assert (
                stored_langchain_documents[0].metadata["mlrun_target_path"]
                == stored_artifcat.get_target_path()
            )

    def test_vectorstore_document_mlrun_artifact(self):
        # Check mlrun loader
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write("This is a test document for vector store.")
            temp_file.flush()
            # Test logging a document localy
            loader = MLRunLoader(
                source_path=temp_file.name,
                loader_spec=DocumentLoaderSpec(),
                artifact_key="test_document_artifact",
                producer=self.project,
                upload=False,
            )
            lc_documents = loader.load()
            assert len(lc_documents) == 1

    def test_directory_loader(self):
        from langchain_community.document_loaders import (
            DirectoryLoader,
        )

        temp_dir = tempfile.mkdtemp()

        # Create txt file
        with open(os.path.join(temp_dir, "sample1.txt"), "w") as f:
            f.write(
                "This is a sample text file.\nIt contains multiple lines.\nFor testing purposes."
            )

        with open(os.path.join(temp_dir, "sample2.txt"), "w") as f:
            f.write(
                "This is a sample text file2.\nIt contains multiple lines.\nFor testing purposes."
            )

        artifact_loader_spec = DocumentLoaderSpec(
            loader_class_name="langchain_community.document_loaders.UnstructuredFileLoader",
            src_name="file_path",
            kwargs={},
        )
        loader = DirectoryLoader(
            temp_dir,
            glob="**/*.*",
            loader_cls=MLRunLoader,
            kwargs={
                "source_path": temp_dir.name,
                "loader_spec": artifact_loader_spec,
                "artifact_key": "doc%%",
                "producer": self.project,
                "upload": False,
            },
        )
        documents = loader.load()
        print(documents)

    def test_chroma_db(self):
        from langchain.embeddings import FakeEmbeddings

        embedding_model = FakeEmbeddings(size=3)
        profile = VectorStoreProfile(
            name="chroma",
            vector_store_class="langchain_community.vectorstores.Chroma",
            kwargs_private={
                "client_settings": {
                    "chroma_server_host": "192.168.226.201",
                    "chroma_server_http_port": 8000,
                    "is_persistent": True,
                }
            },
        )
        register_temporary_client_datastore_profile(profile)
        self.project.register_datastore_profile(profile)

        self.project.get_or_create_vector_store_collection(
            collection_name="collection_name",
            profile=profile.name,
            embedding_function=embedding_model,
        )

    def test_vectorstore_collection_documents(self):
        from langchain.embeddings import FakeEmbeddings

        embedding_model = FakeEmbeddings(size=3)
        profile = VectorStoreProfile(
            name="milvus",
            vector_store_class="langchain_community.vectorstores.Milvus",
            kwargs_private={
                "connection_args": {
                    "host": config["MILVUS_HOST"],
                    "port": config["MILVUS_PORT"],
                }
            },
        )
        collection = self.project.get_or_create_vector_store_collection(
            collection_name="collection_name",
            profile=profile,
            embedding_function=embedding_model,
            auto_id=True,
        )
        with tempfile.NamedTemporaryFile(mode="w") as temp_file1:
            temp_file1.write(
                "Machine learning enables computers to learn from data without being explicitly programmed."
            )
            temp_file1.flush()
            with tempfile.NamedTemporaryFile(mode="w") as temp_file2:
                temp_file2.write(
                    "Machine learning enables computers to learn from data without being explicitly programmed."
                )
                temp_file2.flush()
                with tempfile.NamedTemporaryFile(mode="w") as temp_file3:
                    temp_file3.write(
                        "Machine learning enables computers to learn from data without being explicitly programmed."
                    )
                    temp_file3.flush()

                    doc1 = self.project.log_document(
                        "lc_doc1", src_path=temp_file1.name, upload=False
                    )
                    doc2 = self.project.log_document(
                        "lc_doc2", src_path=temp_file2.name, upload=False
                    )
                    doc3 = self.project.log_document(
                        "lc_doc3", src_path=temp_file3.name, upload=False
                    )

                    milvus_ids = collection.add_artifacts([doc1, doc2])
                    assert len(milvus_ids) == 2

                    direct_milvus_id = collection.add_documents(
                        doc3.to_langchain_documents()
                    )
                    assert len(direct_milvus_id) == 1
                    milvus_ids.append(direct_milvus_id[0])

                    collection.col.flush()
                    documents_in_db = collection.similarity_search(
                        query="",
                        expr=f"{doc1.METADATA_ORIGINAL_SOURCE_KEY} == '{temp_file1.name}'",
                    )
                    assert len(documents_in_db) == 1

                    collection.delete_artifacts([doc1])
                    collection.col.flush()

                    documents_in_db = collection.similarity_search(
                        query="",
                        expr=f"{doc1.METADATA_ORIGINAL_SOURCE_KEY} == '{temp_file1.name}'",
                    )
                    assert len(documents_in_db) == 0
            # collection.remove_from_artifact(doc1)
            # assert something

        collection.col.drop()
