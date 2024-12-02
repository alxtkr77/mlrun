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
import random
import string
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
def generate_random_text(length: int) -> str:
    chars = string.ascii_letters + string.digits + " "
    return "".join(random.choice(chars) for _ in range(length))


# Marked as enterprise because of v3io mount and pipelines
@TestMLRunSystem.skip_test_if_env_not_configured
@pytest.mark.enterprise
class TestDatastoreProfile(TestMLRunSystem):
    def custom_setup(self):
        pass

    def test_vectorstore_document_artifact(self):
        sample_content = generate_random_text(1000)
        artifact_key = "test_document_artifact"
        # Create a temporary text file with a simple context
        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write(sample_content)
            temp_file.flush()
            # Test logging a document localy
            artifact = self.project.log_document(
                artifact_key, local_path=temp_file.name, upload=False
            )
            langchain_documents = artifact.to_langchain_documents()

            assert len(langchain_documents) == 1
            assert langchain_documents[0].page_content == sample_content
            assert (
                langchain_documents[0].metadata["source"]
                == f"{self.project.name}/{artifact_key}"
            )
            assert langchain_documents[0].metadata["original_source"] == temp_file.name
            assert langchain_documents[0].metadata["mlrun_object_uri"] == artifact.uri
            assert langchain_documents[0].metadata["mlrun_chunk"] == "0"

            # Test logging a document localy
            artifact = self.project.log_document(
                artifact_key, local_path=temp_file.name, upload=True
            )

            stored_artifcat = self.project.get_artifact(artifact_key)
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

    def test_vectorstore_loader(self):
        sample_content = generate_random_text(1000)
        artifact_key = "test_document_artifact"

        with tempfile.NamedTemporaryFile(mode="w") as temp_file:
            temp_file.write(sample_content)
            temp_file.flush()

            loader = MLRunLoader(
                source_path=temp_file.name,
                loader_spec=DocumentLoaderSpec(),
                artifact_key=artifact_key,
                producer=self.project,
            )
            lc_documents = loader.load()
            assert len(lc_documents) == 1
            assert lc_documents[0].page_content == sample_content

            stored_artifcat = self.project.get_artifact(artifact_key)
            lc_documents = stored_artifcat.to_langchain_documents()
            assert len(lc_documents) == 1
            assert lc_documents[0].page_content == sample_content

    def test_directory_loader(self):
        from langchain_community.document_loaders import (
            DirectoryLoader,
        )

        temp_dir = tempfile.mkdtemp()
        sample_content1 = generate_random_text(1000)
        sample_content2 = generate_random_text(1000)

        artifact_key = "doc%%"
        artifact_key1 = MLRunLoader.artifact_key_instance(
            artifact_key, f"{temp_dir}/sample1.txt"
        )
        artifact_key2 = MLRunLoader.artifact_key_instance(
            artifact_key, f"{temp_dir}/sample2.txt"
        )

        # Create txt file
        with open(os.path.join(temp_dir, "sample1.txt"), "w") as f:
            f.write(sample_content1)

        with open(os.path.join(temp_dir, "sample2.txt"), "w") as f:
            f.write(sample_content2)

        artifact_loader_spec = DocumentLoaderSpec(
            loader_class_name="langchain_community.document_loaders.TextLoader",
            src_name="file_path",
            kwargs={},
        )
        loader = DirectoryLoader(
            temp_dir,
            glob="**/*.*",
            loader_cls=MLRunLoader,
            loader_kwargs={
                "loader_spec": artifact_loader_spec,
                "artifact_key": artifact_key,
                "producer": self.project,
                "upload": False,
            },
        )

        documents = loader.load()
        if documents[0].metadata["original_source"] == f"{temp_dir}/sample1.txt":
            assert documents[0].page_content == sample_content1
            assert documents[1].page_content == sample_content2
        else:
            assert documents[0].page_content == sample_content2
            assert documents[1].page_content == sample_content1

        stored_artifcat = self.project.get_artifact(artifact_key1)
        lc_documents = stored_artifcat.to_langchain_documents()
        assert lc_documents[0].page_content == sample_content1

        stored_artifcat = self.project.get_artifact(artifact_key2)
        lc_documents = stored_artifcat.to_langchain_documents()
        assert lc_documents[0].page_content == sample_content2

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

        register_temporary_client_datastore_profile(profile)
        self.project.register_datastore_profile(profile)

        collection = self.project.get_or_create_vector_store_collection(
            collection_name="collection_name",
            profile=profile.name,
            embedding_function=embedding_model,
            auto_id=True,
        )
        with tempfile.NamedTemporaryFile(mode="w") as temp_file1:
            temp_file1.write(generate_random_text(1000))
            temp_file1.flush()
            with tempfile.NamedTemporaryFile(mode="w") as temp_file2:
                temp_file2.write(generate_random_text(1000))
                temp_file2.flush()
                with tempfile.NamedTemporaryFile(mode="w") as temp_file3:
                    temp_file3.write(generate_random_text(1000))
                    temp_file3.flush()

                    doc1 = self.project.log_document(
                        "lc_doc1", local_path=temp_file1.name, upload=False
                    )
                    doc2 = self.project.log_document(
                        "lc_doc2", local_path=temp_file2.name, upload=False
                    )
                    doc3 = self.project.log_document(
                        "lc_doc3", local_path=temp_file3.name, upload=False
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

        collection.remove_from_artifact(doc2)
        assert len(doc2.spec.collections) == 0

        doc3 = self.project.get_artifact("lc_doc3")
        doc3.collection_remove(collection.id)
        self.project.update_artifact(doc3)

        doc3 = self.project.get_artifact("lc_doc3")
        assert len(doc3.spec.collections) == 0

        collection.col.drop()

    def test_vectorstore_splitter_and_ids(self):
        from langchain.embeddings import FakeEmbeddings
        from langchain.text_splitter import CharacterTextSplitter

        splitter = CharacterTextSplitter(
            separator="",  # Empty string means split by pure character count
            chunk_size=100,  # Each chunk will be exactly 100 characters
            chunk_overlap=0,  # No overlap between chunks
        )

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
            collection_name="collection_name_with_ids",
            profile=profile,
            embedding_function=embedding_model,
        )
        with tempfile.NamedTemporaryFile(mode="w") as temp_file1:
            temp_file1.write(generate_random_text(200))
            temp_file1.flush()
            doc1 = self.project.log_document("lc_doc1", local_path=temp_file1.name)
            doc2 = self.project.log_document("lc_doc2", local_path=temp_file1.name)

            ids = collection.add_artifacts(
                [doc1, doc2], splitter=splitter, ids=["1", "2"]
            )
            assert ids == ["1_1", "1_2", "2_1", "2_2"]

            ids = collection.add_artifacts([doc1], ids=["3"])
            assert ids == ["3"]

            res = doc1.to_langchain_documents()
            ids = collection.add_documents(res, ids=["123"])
            assert ids == ["123"]

        collection.col.drop()
