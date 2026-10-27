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
