# Streamlit과 Ollama로 만드는 로컬 AI 채팅

두 개의 Python 예제로 Streamlit의 입력 처리와 채팅을 실습한다. 첫 예제는 이름을 입력받아 인사말을 출력한다. 두 번째 예제는 로컬 Ollama에서 `qwen3.5:2b`를 실행하고, 앞선 대화를 바탕으로 생성한 한국어 답변을 화면에 실시간으로 표시한다.

## 준비

Python 3.12와 `uv`를 사용한다. 저장소를 내려받은 뒤 프로젝트 루트에서 의존성을 설치한다.

```powershell
uv sync --locked
```

## 예제 1 · 첫 Streamlit 앱

```powershell
uv run streamlit run src/ex01/app.py
```

브라우저에서 이름을 입력하고 **인사 출력**을 누르면 입력값이 인사말에 반영된다. 이 예제는 Ollama 없이 실행할 수 있다. [예제 1 코드](src/ex01/app.py)

## 예제 2 · 이전 대화를 기억하는 AI 채팅

Ollama를 실행한 뒤 다음 명령을 사용한다. `ollama pull`은 모델을 처음 준비할 때 한 번 실행하면 된다.

```powershell
ollama pull qwen3.5:2b
uv run streamlit run src/ex02/app.py
```

같은 채팅에서 “딸기가 3개 있습니다”, “저와 친구가 1개씩 먹었습니다”, “몇 개 남았나요?”를 차례로 보내 보자. 마지막 질문의 기대 답은 **1개**다. 앱은 매 요청에 앞선 대화를 함께 보내므로 모델은 처음 수량과 두 사람이 먹은 양을 모두 참조할 수 있다.

사이드바의 **최대 추론 글자 수**는 기본 2,048자다. 답변이 나오기 전에 추론이 한도에 닿거나 같은 문장이 반복되면, 앱은 수집한 추론을 담아 답변을 이어 쓰도록 요청한다. 이때도 `reasoning=True`를 유지한다. 반복 판정 기준과 터미널 로그를 읽는 방법은 [예제 2 실습 안내](src/ex02/readme.md)에 설명돼 있다.

## 테스트

다음 명령은 모델 응답을 모의해 대화 기록 전달, 추론 반복 감지, 빈 답변 처리 등을 검사한다. Ollama 서버를 실행하지 않아도 된다.

```powershell
uv run python -m unittest discover -s tests -v
```
