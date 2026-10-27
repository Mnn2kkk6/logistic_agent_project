import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from src import rag


class RAGValidationTests(unittest.TestCase):
    def test_validate_query_trims_whitespace(self):
        self.assertEqual(rag.validate_query("  What is late risk?  ", "4"),
                         ("What is late risk?", 4))

    def test_validate_query_rejects_empty_and_oversized_questions(self):
        with self.assertRaises(ValueError):
            rag.validate_query("   ")
        with self.assertRaises(ValueError):
            rag.validate_query("x" * 2001)

    def test_validate_query_rejects_out_of_range_top_k(self):
        for value in (0, 9, "not-a-number"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                rag.validate_query("valid question", value)

    def test_candidate_sources_exclude_raw_order_csv(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("docs", encoding="utf-8")
            (root / "data_2").mkdir()
            (root / "data_2" / "DescriptionDataCoSupplyChain.csv").write_text(
                "field,description\\nlate_delivery_risk,Existing source label\\n",
                encoding="utf-8",
            )
            (root / "data_2" / "DataCoSupplyChainDataset.csv").write_text(
                "sensitive,raw\\n", encoding="utf-8"
            )
            sources = rag._candidate_source_paths(root)
            names = {path.name for path in sources}
            self.assertIn("README.md", names)
            self.assertIn("DescriptionDataCoSupplyChain.csv", names)
            self.assertNotIn("DataCoSupplyChainDataset.csv", names)

    def test_document_point_id_is_a_deterministic_uuid(self):
        document = type("Doc", (), {
            "metadata": {"source": "README.md", "chunk_index": 3},
            "page_content": "A sample chunk",
        })()
        first = rag._document_point_id(document)
        second = rag._document_point_id(document)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 36)
        self.assertEqual(first.count("-"), 4)

    def test_qdrant_vector_store_accepts_deterministic_point_ids(self):
        from langchain_core.documents import Document
        from langchain_core.embeddings import Embeddings
        from langchain_qdrant import QdrantVectorStore
        from qdrant_client import QdrantClient, models

        class TinyEmbeddings(Embeddings):
            @staticmethod
            def _vector(text):
                lowered = text.lower()
                return [
                    float("delivery" in lowered),
                    float("late" in lowered),
                    float("warehouse" in lowered),
                    1.0,
                ]

            def embed_documents(self, texts):
                return [self._vector(text) for text in texts]

            def embed_query(self, text):
                return self._vector(text)

        client = QdrantClient(":memory:")
        collection = "rag_unit_test"
        client.create_collection(
            collection_name=collection,
            vectors_config=models.VectorParams(
                size=4,
                distance=models.Distance.COSINE,
            ),
        )
        store = QdrantVectorStore(
            client=client,
            collection_name=collection,
            embedding=TinyEmbeddings(),
        )
        documents = [
            Document(
                page_content="Check late delivery events.",
                metadata={"source": "sop.md", "chunk_index": 0},
            ),
            Document(
                page_content="Warehouse capacity planning.",
                metadata={"source": "warehouse.md", "chunk_index": 1},
            ),
        ]
        ids = [rag._document_point_id(doc) for doc in documents]
        store.add_documents(documents=documents, ids=ids)
        matches = store.similarity_search_with_score("late delivery", k=1)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0][0].metadata["source"], "sop.md")
        client.close()

    def test_answer_abstains_when_retrieval_is_below_threshold(self):
        weak_result = [{
            "page_content": "unrelated text",
            "source": "knowledge_base/example.md",
            "title": "example",
            "score": 0.05,
        }]
        with patch.object(rag, "search_documents", return_value=weak_result), \
             patch.object(rag, "_generate_grounded_answer") as generate:
            result = rag.answer_question("What is the return policy?")
        self.assertTrue(result["abstained"])
        self.assertEqual(result["sources"], [])
        generate.assert_not_called()

    def test_answer_passes_retrieved_evidence_and_returns_sources(self):
        retrieved = [{
            "page_content": "late_delivery_risk is an existing label in the dataset.",
            "source": "data_2/README.md",
            "title": "README",
            "score": 0.72,
            "chunk_index": 2,
        }]
        with patch.object(rag, "search_documents", return_value=retrieved), \
             patch.object(rag, "_generate_grounded_answer", return_value="It is an existing label [S1].") as generate:
            result = rag.answer_question("What does late_delivery_risk mean?")
        self.assertFalse(result["abstained"])
        self.assertEqual(result["sources"][0]["id"], "S1")
        self.assertEqual(result["sources"][0]["source"], "data_2/README.md")
        self.assertIn("[S1]", generate.call_args.args[1])
        self.assertEqual(result["answer"], "It is an existing label [S1].")


if __name__ == "__main__":
    unittest.main()
