"""채팅과 문서 검색 앱에서 사용하는 모델·검색 함수."""

import logging
import os
import re
from collections import Counter
from pathlib import Path
from time import perf_counter
from uuid import uuid4

import streamlit as st

MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:2b")
HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
EMBED_MODEL = os.getenv("HF_EMBED_MODEL", "intfloat/multilingual-e5-small")
DOCUMENT = Path(__file__).parent / "data/library.txt"
REASONING_CHAR_LIMIT = 2048
_SENTENCE_END = re.compile(r"[.!?。！？\n]+")

logger = logging.getLogger(__name__)
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s/%(request_id)s] %(message)s",
        defaults={"request_id": "-"},
    ))
    logger.addHandler(handler)
logger.setLevel(logging.DEBUG if os.getenv("OLLAMA_LOG_LEVEL", "INFO").upper() == "DEBUG" else logging.INFO)
logger.propagate = False


def prepare():
    from sentence_transformers import SentenceTransformer

    SentenceTransformer(EMBED_MODEL)
    print("임베딩 모델 준비 완료:", EMBED_MODEL)


class EmptyAnswerError(RuntimeError):
    """모델이 표시할 답변 없이 생성을 끝냈습니다."""


def _repeats_reasoning(text):
    # 스트리밍 중인 마지막 문장은 아직 바뀔 수 있어, 문장 부호나 줄바꿈으로 끝난 부분만 비교한다.
    sentences = (" ".join(s.casefold().split()) for s in _SENTENCE_END.split(text)[:-1])
    counts = Counter(s for s in sentences if len(s) >= 20)
    return any(count >= 3 for count in counts.values())


def _stream_attempt(model, chat, char_limit, log, *, completing=False):
    """한 번의 모델 호출에서 답변 조각을 전달하고, 종료 시 답변 유무와 수집한 추론을 반환한다."""
    started = perf_counter()
    chunks = 0
    done_reason = None
    eval_count = None
    reasoning_text = ""
    answer_parts = []
    has_answer = False
    options = {"temperature": 0} if completing else None
    stream = model.stream(chat, reasoning=True, options=options)
    try:
        for part in stream:
            chunks += 1
            # ChatOllama는 추론과 답변 본문을 서로 다른 필드에 담는다.
            thinking = part.additional_kwargs.get("reasoning_content")
            content = part.text
            done_reason = part.response_metadata.get("done_reason", done_reason)
            eval_count = part.response_metadata.get("eval_count", eval_count)
            log.debug(
                "청크 %d: reasoning_content=%r content=%r metadata=%r",
                chunks, thinking, part.content, part.response_metadata,
            )
            if thinking:
                if not reasoning_text:
                    log.info("추론 수신 시작 (%.2f초)", perf_counter() - started)
                reasoning_text += thinking[:char_limit - len(reasoning_text)]
            if content:
                answer_parts.append(content)
                if has_answer:
                    yield content
                elif content.strip():
                    # 본문이 시작된 뒤에만 첫 출력을 내보내되, 앞서 받은 들여쓰기와 줄바꿈은 살린다.
                    has_answer = True
                    log.info("답변 수신 시작 (%.2f초)", perf_counter() - started)
                    yield "".join(answer_parts)
            # 같은 청크에 답변까지 도착했으면 위에서 시작한 답변을 끝까지 받는다.
            if thinking and not has_answer:
                if len(reasoning_text) >= char_limit:
                    done_reason = "reasoning_char_limit"
                    break
                if _SENTENCE_END.search(thinking) and _repeats_reasoning(reasoning_text):
                    done_reason = "repeated_reasoning"
                    break
        return has_answer, reasoning_text
    finally:
        close = getattr(stream, "close", None)
        if close is not None:
            close()
        answer_text = "".join(answer_parts)
        log.info(
            "수신 내용 (reasoning=True, %.2f초, 청크=%d, done_reason=%s, eval_count=%s)"
            "\n추론 원문 (%d자):\n%s\n답변 원문 (%d자):\n%s",
            perf_counter() - started, chunks, done_reason, eval_count,
            len(reasoning_text), reasoning_text or "(없음)",
            len(answer_text), answer_text or "(없음)",
        )


def stream_text(messages, *, reasoning_char_limit=None):
    """현재 대화의 답변을 스트리밍하고, 답변이 비었을 때만 모델에 한 번 더 요청한다."""
    from langchain_core.messages import AIMessage, HumanMessage, convert_to_messages
    from langchain_ollama import ChatOllama

    log = logging.LoggerAdapter(logger, {"request_id": uuid4().hex[:8]})
    limit = REASONING_CHAR_LIMIT if reasoning_char_limit is None else reasoning_char_limit
    if limit <= 0:
        raise ValueError("추론 글자 수는 양수여야 합니다.")
    try:
        chat = convert_to_messages([messages] if isinstance(messages, str) else messages)
        question = next((m.text for m in reversed(chat) if isinstance(m, HumanMessage)), "")
        log.info("요청 시작: model=%s reasoning=True\n질문:\n%s", MODEL, question or "(없음)")
        model = ChatOllama(model=MODEL, base_url=HOST)
        has_answer, thought = yield from _stream_attempt(model, chat, limit, log)
        if has_answer:
            return
        if thought:
            # 두 번째 요청도 reasoning=True다. Qwen은 전달받은 추론 뒤에 </think>를 붙여 본문부터 이어 쓴다.
            # 줄바꿈은 Ollama 파서에 본문 시작을 알리고 렌더링할 때 제거된다.
            chat = [*chat, AIMessage(content="\n", additional_kwargs={"reasoning_content": thought})]
            log.info("수집한 추론 %d자를 사용해 같은 턴의 답변을 이어 씁니다.", len(thought))
        else:
            log.warning("빈 응답을 받아 추론을 켠 채 한 번 다시 요청합니다.")
        # 두 요청은 한 질문에 대한 처리이므로 이미 수집한 글자 수만큼 추론 한도에서 뺀다.
        has_answer, _ = yield from _stream_attempt(model, chat, limit - len(thought), log, completing=bool(thought))
        if has_answer:
            return
    except Exception:
        log.exception("응답 처리 실패: model=%s", MODEL)
        raise

    log.error("재시도 후에도 최종 답변이 비어 있습니다.")
    raise EmptyAnswerError("모델이 추론 후 최종 답변을 완성하지 못했습니다.")


@st.cache_resource
def embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBED_MODEL, local_files_only=True)


@st.cache_data
def document_vectors(chunks):
    return embedder().encode(["passage: " + chunk for chunk in chunks], normalize_embeddings=True)


def retrieve(text, question):
    chunks = [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]
    if not chunks:
        return []
    vectors = document_vectors(chunks)
    query = embedder().encode(["query: " + question], normalize_embeddings=True)[0]
    scores = vectors @ query
    return [(index + 1, float(scores[index]), chunks[index]) for index in scores.argsort()[::-1][:3]]
