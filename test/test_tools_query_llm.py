import unittest
from unittest.mock import Mock, patch

from odak.tools.large_language_model import query_llm


class TestQueryLLM(unittest.TestCase):
    @patch("odak.tools.large_language_model.requests.post")
    def test_query_llm_returns_response(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": "response"}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = query_llm("Hello")

        self.assertEqual(response, "response")

    @patch("odak.tools.large_language_model.requests.post")
    def test_query_llm_strips_response(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": "  response  \n"}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        response = query_llm("Hello")

        self.assertEqual(response, "response")

    @patch("odak.tools.large_language_model.requests.post")
    def test_query_llm_uses_custom_parameters(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {"message": {"content": "response"}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        query_llm(
            "Hello",
            address="1.2.3.4",
            port=9999,
            model="test-model",
            endpoint="/api/chat",
            temperature=0.1,
            max_tokens=20,
            timeout=5,
        )

        args, kwargs = mock_post.call_args

        self.assertEqual(args[0], "http://1.2.3.4:9999/api/chat")
        self.assertEqual(kwargs["timeout"], 5)
        self.assertEqual(kwargs["json"]["model"], "test-model")
        self.assertEqual(kwargs["json"]["options"]["temperature"], 0.1)
        self.assertEqual(kwargs["json"]["options"]["num_predict"], 20)

    @patch("odak.tools.large_language_model.requests.post")
    def test_query_llm_missing_message_raises_value_error(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with self.assertRaises(ValueError):
            query_llm("Hello")

    @patch("odak.tools.large_language_model.requests.post")
    def test_query_llm_missing_content_raises_value_error(self, mock_post):
        mock_response = Mock()
        mock_response.json.return_value = {"message": {}}
        mock_response.raise_for_status.return_value = None
        mock_post.return_value = mock_response

        with self.assertRaises(ValueError):
            query_llm("Hello")

    def test_query_llm_requires_string_prompt(self):
        with self.assertRaises(TypeError):
            query_llm(123)


if __name__ == "__main__":
    unittest.main()