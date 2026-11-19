import streamlit as st

from shared import REASONING_CHAR_LIMIT, EmptyAnswerError, stream_text

st.title("로컬 AI 대화")
# Streamlit은 입력할 때마다 스크립트를 다시 실행하므로 이전 대화를 세션에 보관한다.
if "messages" not in st.session_state:
    st.session_state.messages = []
if st.sidebar.button("대화 지우기"):
    st.session_state.messages = []
reasoning_limit = st.sidebar.number_input(
    "최대 추론 글자 수", min_value=256, max_value=8192, value=REASONING_CHAR_LIMIT, step=256,
    help="한도에 도달하면 그때까지의 추론을 바탕으로 답변을 이어 씁니다.",
)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if question := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)
    system_prompt = (
        "사용자에게 한국어로 짧게 답한다. 질문이나 요청 없이 사실만 알려 주면 '알겠습니다.'라고 확인한다. "
        "질문이나 요청을 받으면 이전 사용자 메시지에서 알려 준 사실을 시간순으로 반영한다. "
        "수량 질문에는 시작 수량과 변화를 실제 숫자로 대입한 계산식, 결과, 단위를 쓴다."
    )
    # 모델은 매 요청마다 이 목록을 읽는다. 앞선 사용자 발언도 순서대로 보내야 후속 질문을 풀 수 있다.
    messages = [{"role": "system", "content": system_prompt}, *st.session_state.messages]
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_text(messages, reasoning_char_limit=reasoning_limit))
        except EmptyAnswerError as error:
            st.error(str(error))
        else:
            # 세션에는 화면에 표시한 완성 답변을 한 번만 추가한다.
            st.session_state.messages.append({"role": "assistant", "content": answer})
