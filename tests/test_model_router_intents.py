import unittest
from unittest.mock import patch

from src import model_router, providers


class DirectRAGRoutingTests(unittest.TestCase):
    @patch("src.model_router.rag.answer_question")
    @patch("src.model_router.providers.chat_turn")
    def test_document_question_bypasses_agent_tool_selection(self, chat_turn, answer_question):
        answer_question.return_value = {
            "answer": "DataCo dùng nhãn có sẵn [S1].",
            "sources": [
                {"id": "S1", "title": "DataCo README", "source": "data_2/README.md"}
            ],
            "retrieved_count": 1,
            "abstained": False,
        }

        reply, tool_calls, new_history, route = model_router.chat_turn_auto(
            [], "Theo README, giải thích kiến trúc dự án", dataset_id="dataco"
        )

        answer_question.assert_called_once()
        chat_turn.assert_not_called()
        self.assertIn("[S1]", reply)
        self.assertIn("data_2/README.md", reply)
        self.assertEqual(tool_calls, [])
        self.assertEqual(route["intent"], "rag")
        self.assertEqual(route["model_id"], "direct-rag")
        self.assertEqual(new_history[-1]["content"], reply)

    @patch("src.model_router.rag.answer_question", side_effect=model_router.rag.RAGNotIndexedError("not indexed"))
    @patch("src.model_router.providers.chat_turn")
    def test_rag_unavailable_returns_clear_chat_message(self, chat_turn, answer_question):
        reply, _, history, route = model_router.chat_turn_auto(
            [], "Theo README, giải thích kiến trúc dự án", dataset_id="olist"
        )
        answer_question.assert_called_once()
        chat_turn.assert_not_called()
        self.assertIn("chưa thể tra cứu kho tài liệu", reply.lower())
        self.assertEqual(history[-1]["content"], reply)
        self.assertTrue(route["fallback_used"])


class IntentToolFilteringTests(unittest.TestCase):
    def _tools_for(self, intent, dataset="olist"):
        dataset_token = providers._ACTIVE_DATASET_ID.set(dataset)
        intent_token = providers._ACTIVE_INTENT.set(intent)
        try:
            return providers._active_tools()
        finally:
            providers._ACTIVE_INTENT.reset(intent_token)
            providers._ACTIVE_DATASET_ID.reset(dataset_token)

    def test_data_route_hides_knowledge_base_tool(self):
        names = {tool["name"] for tool in self._tools_for("data_ml")}
        self.assertNotIn("query_knowledge_base", names)
        self.assertIn("query_dataset_sql", names)

    def test_general_route_has_no_tools(self):
        self.assertEqual(self._tools_for("general"), [])

    def test_hybrid_route_keeps_knowledge_and_dataset_tools(self):
        names = {tool["name"] for tool in self._tools_for("hybrid", "dataco")}
        self.assertIn("query_knowledge_base", names)
        self.assertIn("query_dataset_sql", names)


if __name__ == "__main__":
    unittest.main()
