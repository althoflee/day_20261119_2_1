import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage, AIMessageChunk

import shared


class StreamTextTests(unittest.TestCase):
    def setUp(self):
        self.messages = [
            {"role": "system", "content": "한국어로 짧게 답한다."},
            {"role": "user", "content": "접시에 딸기가 3개있습니다"},
        ]
        self.stream = self.enterContext(patch("langchain_ollama.ChatOllama")).return_value.stream
        self.logs = self.enterContext(self.assertLogs(shared.logger, level="INFO"))
        self.enterContext(patch.object(shared, "MODEL", "qwen3.5:2b"))

    def assert_continued_with(self, reasoning):
        first, continuation = self.stream.call_args_list
        self.assertIs(first.kwargs["reasoning"], True)
        self.assertIs(continuation.kwargs["reasoning"], True)
        history = continuation.args[0]
        self.assertEqual(history[:-1], first.args[0])
        self.assertIsInstance(history[-1], AIMessage)
        self.assertEqual(history[-1].content, "\n")
        self.assertEqual(history[-1].additional_kwargs["reasoning_content"], reasoning)

    def test_reasoning_only_response_is_continued_without_restarting_the_reasoning(self):
        self.stream.side_effect = [
            [AIMessageChunk(
                content=" \n",
                additional_kwargs={"reasoning_content": "Final: 아직 후보인 답변"},
                response_metadata={"done_reason": "length"},
            )],
            [AIMessageChunk(content="네, "), AIMessageChunk(content="딸기는 3개입니다.")],
        ]

        answer = "".join(shared.stream_text(self.messages))

        self.assertEqual(answer, "네, 딸기는 3개입니다.")
        self.assert_continued_with("Final: 아직 후보인 답변")
        transcript = "\n".join(self.logs.output)
        self.assertIn("Final: 아직 후보인 답변", transcript)
        self.assertIn(answer, transcript)

    def test_character_limit_closes_the_stream_before_requesting_the_answer(self):
        self.enterContext(patch.object(shared, "REASONING_CHAR_LIMIT", 24))
        thought = "The plate already contains three strawberries."
        closed = []

        def first_stream():
            try:
                yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": thought})
                self.fail("the model was asked for another chunk after the reasoning limit")
            finally:
                closed.append(True)

        running = first_stream()
        self.addCleanup(running.close)

        def requests(chat, **kwargs):
            if len(chat) == len(self.messages):
                return running
            self.assertEqual(closed, [True])
            return iter([AIMessageChunk(content="딸기는 세 개예요.")])

        self.stream.side_effect = requests
        self.assertEqual("".join(shared.stream_text(self.messages)), "딸기는 세 개예요.")
        self.assert_continued_with(thought[:24])

    def test_repeated_sentence_split_between_chunks_stops_before_the_character_limit(self):
        sentence = "I should confirm that the plate already contains three strawberries.\n"

        def looping():
            for _ in range(3):
                yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": sentence[:15]})
                yield AIMessageChunk(content="", additional_kwargs={"reasoning_content": sentence[15:]})
            self.fail("a fourth repetition was requested")

        self.stream.side_effect = [looping(), [AIMessageChunk(content="네, 세 개예요.")]]
        self.assertEqual("".join(shared.stream_text(self.messages)), "네, 세 개예요.")
        self.assert_continued_with(sentence * 3)

    def test_real_answer_in_same_chunk_as_reasoning_limit_is_returned_directly(self):
        self.enterContext(patch.object(shared, "REASONING_CHAR_LIMIT", 16))
        self.stream.return_value = [
            AIMessageChunk(content="있는 그대로 ", additional_kwargs={"reasoning_content": "a" * 32}),
            AIMessageChunk(content="답합니다."),
        ]
        self.assertEqual("".join(shared.stream_text(self.messages)), "있는 그대로 답합니다.")
        self.stream.assert_called_once()

    def test_valid_answer_preserves_whitespace_and_does_not_retry(self):
        self.stream.return_value = [
            AIMessageChunk(content="\n"), AIMessageChunk(content="  답"), AIMessageChunk(content="\n"),
        ]

        self.assertEqual("".join(shared.stream_text(self.messages)), "\n  답\n")
        self.stream.assert_called_once()

    def test_both_attempts_empty_raise_instead_of_returning_a_blank_reply(self):
        self.stream.side_effect = [[AIMessageChunk(content="")], [AIMessageChunk(content=" \n")]]

        with self.assertRaises(shared.EmptyAnswerError):
            next(shared.stream_text(self.messages))
        self.assertEqual(self.stream.call_count, 2)
        self.assertTrue(all(call.kwargs["reasoning"] is True for call in self.stream.call_args_list))

    def test_interrupted_answer_is_not_replayed(self):
        def interrupted():
            yield AIMessageChunk(content="부분 답변")
            raise ConnectionError("stream interrupted")

        self.stream.return_value = interrupted()
        result = shared.stream_text(self.messages)

        self.assertEqual(next(result), "부분 답변")
        with self.assertRaises(ConnectionError):
            next(result)
        self.stream.assert_called_once()

    def test_followup_keeps_prior_facts_and_only_uses_current_thought_for_continuation(self):
        from streamlit.testing.v1 import AppTest

        first = "딸기가 3개있습니다"
        second = "저와 친구가 1개씩 먹었습니다."
        third = "몇개 남았나요"
        thought = "이전 두 턴에서 처음 수량과 먹은 수량을 확인한다."
        self.stream.side_effect = [
            [AIMessageChunk(content="확인했습니다.", additional_kwargs={"reasoning_content": "첫 턴 추론"})],
            [AIMessageChunk(content="네.")],
            [AIMessageChunk(content="", additional_kwargs={"reasoning_content": thought})],
            [AIMessageChunk(content="3 - 1 - 1 = 1개입니다.")],
        ]
        path = Path(__file__).resolve().parents[1] / "src/ex02/app.py"
        app = AppTest.from_file(str(path)).run()
        for message in (first, second, third):
            app.chat_input[0].set_value(message).run()
            self.assertFalse(app.exception)
        history = self.stream.call_args_list[-1].args[0]
        self.assertEqual([m.text for m in history[1:-1]], [first, "확인했습니다.", second, "네.", third])
        self.assertTrue(all("reasoning_content" not in m.additional_kwargs for m in history[:-1]))
        self.assertEqual(history[-1].additional_kwargs["reasoning_content"], thought)
        self.assertEqual(app.session_state["messages"][-1]["content"], app.chat_message[-1].markdown[-1].value)
