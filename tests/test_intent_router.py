import unittest

from src.intent_router import classify_intent


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


if __class__ == "__main__":
    unittest.main()
