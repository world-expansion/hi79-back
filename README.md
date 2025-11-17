# 👫 칭구야(Chingoo-ya)

**CBT·행동활성화 기반 감정 회복 지원 RAG 챗봇(API + 서비스)**

> “당신의 감정을 이해하고, 다시 살아갈 힘을 주는 AI 친구”
> 

📌 **2025 새싹 해커톤(SeSAC Hackathon) 출품작**

📌 **팀명: 세계확장**

📌 **구성원: 정수미 · 이재훈 · 이동호 · 조영진 · 김지운**

---

# 📖 프로젝트 개요

- *칭구야(Chingoo-ya)**는

우울감·고립감을 느끼는 청년들이 스스로 감정을 이해하고 회복 행동을 수행하도록 돕는

**CBT 기반 감정 회복 AI 챗봇**이다.

서비스는 다음 두 가지 핵심 개입(Evidence-Based Intervention)을 중심으로 구성된다:

### 🔹 1. **과거 일기 기반 RAG**

- 사용자 과거 감정·행동 기록을 벡터DB(pgvector)에서 검색
- “비슷한 감정을 느꼈던 순간”을 찾아 비교
- 감정 패턴을 재해석할 수 있도록 도움
    
    [세계확장] 칭구야
    

### 🔹 2. **CBT·행동활성화 매뉴얼 기반 RAG**

- 치료 논문·심리학 매뉴얼을 기반으로
    
    사고 재구성 / 행동 활성화 / 자기자비 훈련 과제를 추천
    
    [세계확장] 칭구야
    

이 두 가지 RAG를 기반으로

단순 공감 챗봇을 넘어 **실제 심리치료 원리를 적용한 개입형 챗봇**을 제공한다.

---

# 🎯 서비스 목표

### ✔ 청년 우울·고립감 완화

### ✔ 전문가 상담 접근성 부족 문제 해결

### ✔ 감정 분석 & 행동 제안 기반 실질적 회복 촉진

### ✔ 감정 패턴 기록 및 시각화를 통한 자기 이해

### ✔ CBT 기반 과제 추천

---

# 🌟 주요 기능

## 1) **CBT 기반 감정 대화 구조**

- 공감 (Emotional Reflection)
- 사고 탐색 (Cognitive Exploration)
- 감정 명확화 (Labeling)
- 행동 제안 (Behavioral Suggestion)
- 인지 왜곡 자동 탐지
    
    (흑백논리, 과일반화, 개인화, 파국화 등)
    
    [세계확장] 칭구야
    

## 2) **근거 기반 개입 (Evidence-based Intervention, EBI)**

### 🟦 a. 논문 기반 CBT / 행동활성화 과제 RAG

- 챗봇 내에 CBT  메뉴얼 기반 대화 유도
- 모델을 통해 나온 감정 기반으로 메뉴얼에 따라 과제 추천

### 🟨 b. 과거 일기 기반 RAG

- 사용자 개인 감정 기록에서 유사한 경험 검색
- “그때 무엇을 했는지” 비교 → 회복 전략 안내
- 감정 패턴 인식 & 자각 증진

## 3) 감정 변화 기록/시각화

- 일/주/월 단위 감정 그래프
- 감정 비율 파이차트

---

# 🧠 시스템 구조

```
사용자 입력(대화 전체)
      ↓
감정 분석 모델 (BERT/RoBERTa)
      ↓
인지 왜곡 탐지
      ↓
[RAG] 과거 일기 검색 (pgvector)
[RAG] CBT 논문 기반 검색
      ↓
GPT-4.1이 CBT 대화 생성
      ↓
[RAG] 행동 활성화 과제 추천
      ↓
감정 일기 자동 저장 → 그래프 생성

```

---

# 🧩 기술 스택

### Backend(API)

- **FastAPI**
- **PostgreSQL + pgvector**
- **LangChain RAG Pipeline**
- **OpenAI GPT-4o-mini**

### NLP & ML

- **text-embedding-3-small**
- **감정 라벨링 BERT 모델**

---

# 📦 백엔드 API 설명 (RAG 챗봇 API)

## 🚀 설치 방법

```bash
python -m venv venv
source venv/bin/activate  # mac/Linux
# venv\Scripts\activate   # Windows

pip install -r requirements.txt

```

---

## ⚙ 환경 설정

```bash
cp .env.example .env
# OPENAI_API_KEY=your_api_key_here 입력

```

---

## 📚 PDF 학습 데이터 준비

`data/` 폴더에 PDF 파일들을 넣으면,

초기화 시 자동 학습(RAG 벡터 스토어 생성)된다.

```bash
mkdir -p data
cp ~/Downloads/*.pdf data/

```

---

# ▶ 서버 실행

```bash
python run.py

```

또는:

```bash
uvicorn app.main:app --reload

```

---

# 💬 API 사용법

## 1) 챗봇 초기화

```bash
curl -X POST http://localhost:8000/api/chatbot/initialize

```

예시 응답:

```json
{
  "success": true,
  "message": "챗봇 초기화 완료! 2개의 PDF 파일을 로드했습니다.",
  "data": {
    "document_count": 245,
    "vector_db_path": "./chroma_db",
    "loaded_files": ["support_guide.pdf", "program_info.pdf"]
  }
}

```

---

## 2) 챗봇 상태 확인

```bash
curl http://localhost:8000/api/chatbot/status

```

---

## 3) 질문하기

```bash
curl -X POST http://localhost:8000/api/chatbot/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "우울감을 해소하기 위한 지원 프로그램에는 어떤 것들이 있나요?"}'

```

---

# 📘 API 엔드포인트 요약

| 기능 | 메서드/URL | 설명 |
| --- | --- | --- |
| Health | `GET /api/health` | 서버 상태 확인 |
| 초기화 | `POST /api/chatbot/initialize` | PDF 불러와 RAG DB 생성 |
| 상태조회 | `GET /api/chatbot/status` | 벡터DB/모델 초기화 여부 확인 |
| 질문 | `POST /api/chatbot/chat` | RAG 기반 상담 답변 생성 |

---

# 📑 API 문서

- Swagger: [**http://localhost:8000/docs**](http://localhost:8000/docs)
- ReDoc: [**http://localhost:8000/redoc**](http://localhost:8000/redoc)

---

# 📁 프로젝트 구조

```
h-sw-h-back/
├── app/
│   ├── main.py
│   ├── config.py
│   ├── routers/
│   │   ├── health.py
│   │   └── chatbot.py
│   ├── services/
│   │   └── vector_store.py
│   ├── utils/
│   │   └── pdf_loader.py
│   └── schemas/
│       └── chatbot.py
├── data/
├── chroma_db/
├── .env
├── .env.example
├── requirements.txt
├── run.py
└── README.md

```

---

# 🔄 작동 원리 (RAG Pipeline)

1. PDF 로드
2. 텍스트 청크 분할 (500자 / overlap 100자)
3. OpenAI 임베딩 생성
4. postgreDB 저장
5. 질문 → 벡터 검색 (Top-k)
6. GPT-4o-mini가 CBT 스타일 답변 생성

---

# ✨ 기대효과

### 1) 접근성 높은 정서 지원 도구

### 2) 전문가 상담의 보조자료로 활용 가능

### 3) 청년 고립·우울 문제 완화

### 4) 감정 그래프 기반 자기 이해 증진
