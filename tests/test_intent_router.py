import unittest

from src.intent_router import classify_intent, filter_tools_for_intent


class IntentRouterTests(unittest.TestCase):
    def test_document_question_routes_to_rag(self):
        self.assertEqual(
            classify_intent("Theo README, hãy mô tả kiến trúc của dự án"),
            "rag",
        )

    def test_field_definition_routes_to_rag(self):
        self.assertEqual(
            classify_intent("late_delivery_risk có ý nghĩa gì?"),
            "rag",
        )

    def test_explaining_snake_case_field_routes_to_rag(self):
        self.assertEqual(classify_intent("Explain late_delivery_risk"), "rag")

    def test_quantitative_question_routes_to_data_tools(self):
        self.assertEqual(
            classify_intent("Tính tỷ lệ giao hàng trễ theo từng market"),
            "data_ml",
        )

    def test_combined_document_and_metric_question_routes_to_hybrid(self):
        self.assertEqual(
            classify_intent(
                "Theo tài liệu, giải thích late_delivery_risk và tính tỷ lệ theo market"
            ),
            "hybrid",
        )

    def test_general_question_does_not_trigger_rag(self):
        self.assertEqual(classify_intent("Xin chào, bạn có thể giúp gì?"), "general")

    def test_english_project_architecture_question_routes_to_rag(self):
        self.assertEqual(classify_intent("Explain the project architecture from the README"), "rag")

    def test_english_field_definition_routes_to_rag(self):
        self.assertEqual(classify_intent("What does late_delivery_risk mean?"), "rag")

    def test_named_dataset_definition_routes_to_rag(self):
        self.assertEqual(classify_intent("What is DataCo?"), "rag")

    def test_document_question_about_metrics_stays_on_rag(self):
        self.assertEqual(
            classify_intent("According to the README, which metrics are recorded?"),
            "rag",
        )

    def test_data_question_about_highest_revenue_routes_to_data(self):
        self.assertEqual(
            classify_intent("Which countries have the highest revenue?"),
            "data_ml",
        )

    def test_data_question_about_country_orders_routes_to_data(self):
        self.assertEqual(
            classify_intent("What are the top 5 countries by orders?"),
            "data_ml",
        )

    def test_project_introduction_routes_to_rag(self):
        self.assertEqual(classify_intent("Giới thiệu bộ dữ liệu DataCo"), "rag")

    def test_unrelated_explanation_stays_general(self):
        self.assertEqual(classify_intent("Explain why rain forms"), "general")

    def test_metric_explanation_routes_to_rag(self):
        self.assertEqual(
            classify_intent("How does late delivery risk work?"),
            "rag",
        )

    def test_project_workflow_question_routes_to_rag(self):
        self.assertEqual(classify_intent("How does this project work?"), "rag")

    def test_repo_specific_question_routes_to_rag(self):
        self.assertEqual(classify_intent("What are the RAG limitations in this repository?"), "rag")

    def test_tool_schemas_are_filtered_by_intent(self):
        tools = [
            {"name": "query_dataset_sql"},
            {"name": "query_knowledge_base"},
            {"name": "predict_new_order"},
        ]
        data_tools = filter_tools_for_intent(tools, "data_ml")
        self.assertEqual(
            {tool["name"] for tool in data_tools},
            {"query_dataset_sql", "predict_new_order"},
        )
        self.assertEqual(filter_tools_for_intent(tools, "general"), [])
        self.assertEqual(filter_tools_for_intent(tools, "hybrid"), tools)

    def test_short_follow_up_inherits_document_intent(self):
        history = [
            {"role": "user", "content": "Theo README, hãy giải thích kiến trúc dự án"},
            {"role": "assistant", "content": "Dự án có API và agent..."},
        ]
        self.assertEqual(
            classify_intent("Giải thích thêm", history),
            "rag",
        )

    def test_short_follow_up_inherits_data_intent(self):
        history = [
            {"role": "user", "content": "Tính tỷ lệ giao hàng trễ theo market"},
            {"role": "assistant", "content": "Kết quả là..."},
        ]
        self.assertEqual(classify_intent("Còn theo quý thì sao?", history), "data_ml")


if __name__ == "__main__":
    unittest.main()
