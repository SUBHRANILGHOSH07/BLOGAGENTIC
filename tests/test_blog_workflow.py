import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient
from langchain_groq import ChatGroq
from src.nodes.blog_node import BlogNode
from src.states.blogstate import Blog


class FakeLLM:
    def __init__(self):
        self.translation_prompts = []

    def invoke(self, prompt):
        return SimpleNamespace(content="Original blog")

    def with_structured_output(self, schema, **kwargs):
        def translate(messages):
            self.translation_prompts.append(messages[0].content)
            return {
                "raw": SimpleNamespace(response_metadata={"finish_reason": "stop"}),
                "parsed": Blog(title="Translated title", content="Translated content"),
                "parsing_error": None,
            }

        return SimpleNamespace(invoke=translate)


# Import the actual graph and API without requiring credentials or a live LLM.
with patch("src.llms.groqllm.GroqLLM.get_llm", return_value=FakeLLM()):
    from src.graphs.graph_builder import GraphBuilder, graph
    import app as app_module


class BlogWorkflowTests(unittest.TestCase):
    def test_translation_sends_native_json_schema_without_tools(self):
        llm = ChatGroq(api_key="test-key", model="openai/gpt-oss-20b")
        translated = {"title": "Titre traduit", "content": "Contenu traduit"}
        response = {"choices": [{
            "message": {"role": "assistant", "content": json.dumps(translated)},
            "finish_reason": "stop",
        }]}
        with patch.object(llm.client, "create", return_value=response) as create:
            result = BlogNode(llm).translation({
                "topic": "Ethical AI", "current_language": "french",
                "blog": {"title": "Ethical AI", "content": "A full blog."},
            })
        self.assertEqual(result["blog"], translated)
        payload = create.call_args.kwargs
        self.assertNotIn("tools", payload)
        self.assertNotIn("tool_choice", payload)
        self.assertFalse(payload["stream"])
        self.assertEqual(payload["response_format"]["type"], "json_schema")
        schema = payload["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        self.assertFalse(schema["schema"]["additionalProperties"])
        self.assertEqual(set(schema["schema"]["required"]), {"title", "content"})
        self.assertIn("Ethical AI", payload["messages"][0]["content"])

    def test_translation_rejects_truncated_response(self):
        llm = ChatGroq(api_key="test-key", model="openai/gpt-oss-20b")
        response = {"choices": [{
            "message": {"role": "assistant", "content": '{"title":"Titre","content":"Cut off'},
            "finish_reason": "length",
        }]}
        with patch.object(llm.client, "create", return_value=response):
            with self.assertRaisesRegex(ValueError, "output token limit"):
                BlogNode(llm).translation({
                    "topic": "Ethical AI", "current_language": "french",
                    "blog": {"title": "Ethical AI", "content": "A full blog."},
                })

    def test_exported_graph_accepts_topic_only(self):
        state = graph.invoke({"topic": "Agentic AI"})
        self.assertEqual(state["current_language"], "english")
        self.assertEqual(state["blog"]["content"], "Original blog")

    def test_topic_graph(self):
        state = GraphBuilder(FakeLLM()).setup_graph("topic").invoke(
            {"topic": "Agentic AI"}
        )
        self.assertEqual(state["blog"]["content"], "Original blog")

    def test_language_graph_normalizes_and_returns_translation(self):
        for language in ("french", "hindi", " FRENCH "):
            with self.subTest(language=language):
                llm = FakeLLM()
                state = GraphBuilder(llm).setup_graph("language").invoke(
                    {"topic": "Agentic AI", "current_language": language}
                )
                self.assertEqual(state["blog"], {
                    "title": "Translated title", "content": "Translated content"
                })
                self.assertEqual(len(llm.translation_prompts), 1)
                self.assertIn(language.strip().lower(), llm.translation_prompts[0])
                self.assertIn("Original blog", llm.translation_prompts[0])

    def test_empty_language_skips_translation(self):
        llm = FakeLLM()
        state = GraphBuilder(llm).setup_graph("language").invoke(
            {"topic": "Agentic AI", "current_language": ""}
        )
        self.assertEqual(state["current_language"], "english")
        self.assertEqual(llm.translation_prompts, [])

    def test_unsupported_graph_language_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "current_language must be"):
            GraphBuilder(FakeLLM()).setup_graph("language").invoke(
                {"topic": "Agentic AI", "current_language": "spanish"}
            )

    def test_api_topic_and_language_requests(self):
        for language in (None, "english", "french", "hindi", " FRENCH "):
            with self.subTest(language=language):
                llm = FakeLLM()
                body = {"topic": "Agentic AI"}
                if language is not None:
                    body["language"] = language
                with patch.object(app_module.GroqLLM, "get_llm", return_value=llm):
                    response = TestClient(app_module.app).post("/blogs", json=body)
                self.assertEqual(response.status_code, 200, response.text)
                expected = "Translated content" if language and language.strip().lower() in {"hindi", "french"} else "Original blog"
                self.assertEqual(response.json()["data"]["blog"]["content"], expected)

    def test_api_rejects_invalid_input_before_llm_initialization(self):
        for body in ({}, {"topic": " "}, {"topic": None},
                     {"topic": "AI", "language": "spanish"},
                     {"topic": "AI", "language": None}):
            with self.subTest(body=body):
                with patch.object(app_module.GroqLLM, "get_llm") as get_llm:
                    response = TestClient(app_module.app).post("/blogs", json=body)
                self.assertEqual(response.status_code, 422, response.text)
                get_llm.assert_not_called()


if __name__ == "__main__":
    unittest.main()
