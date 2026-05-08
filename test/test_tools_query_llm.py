import json
import unittest
from unittest.mock import Mock, patch

from odak.tools.large_language_model import query_llm


class TestQueryLLM(unittest.TestCase):
    def mock_urlopen_response(self, data):
        mock_response = Mock()
        mock_response.read.return_value = json.dumps(data).encode("utf-8")
        mock_response.__enter__ = Mock(return_value=mock_response)
        mock_response.__exit__ = Mock(return_value=None)
        return mock_response

    @patch("odak.tools.large_language_model.urllib.request.urlopen")
    def test_query_llm_returns_response(self, mock_urlopen):
        mock_urlopen.return_value = self.mock_urlopen_response(
            {"message": {"content": "response"}}
        )

        response = query_llm("Hello")

        self.assertEqual(response, "response")

    @patch("odak.tools.large_language_model.urllib.request.urlopen")
    def test_query_llm_strips_response(self, mock_urlopen):
        mock_urlopen.return_value = self.mock_urlopen_response(
            {"message": {"content": "  response  \n"}}
        )

        response = query_llm("Hello")

        self.assertEqual(response, "response")

    @patch("odak.tools.large_language_model.urllib.request.urlopen")
    def test_query_llm_uses_custom_parameters(self, mock_urlopen):
        mock_urlopen.return_value = self.mock_urlopen_response(
            {"message": {"content": "response"}}
        )

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

        request = mock_urlopen.call_args[0][0]

        self.assertEqual(request.full_url, "http://1.2.3.4:9999/api/chat")
        self.assertEqual(mock_urlopen.call_args.kwargs["timeout"], 5)

        payload = json.loads(request.data.decode("utf-8"))

        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["options"]["temperature"], 0.1)
        self.assertEqual(payload["options"]["num_predict"], 20)

    @patch("odak.tools.large_language_model.urllib.request.urlopen")
    def test_query_llm_missing_message_raises_value_error(self, mock_urlopen):
        mock_urlopen.return_value = self.mock_urlopen_response({})

        with self.assertRaises(ValueError):
            query_llm("Hello")

    @patch("odak.tools.large_language_model.urllib.request.urlopen")
    def test_query_llm_missing_content_raises_value_error(self, mock_urlopen):
        mock_urlopen.return_value = self.mock_urlopen_response(
            {"message": {}}
        )

        with self.assertRaises(ValueError):
            query_llm("Hello")

    def test_query_llm_requires_string_prompt(self):
        with self.assertRaises(TypeError):
            query_llm(123)


if __name__ == "__main__":
    unittest.main()