from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.chains import RetrievalQA
from typing import Optional
import os

class VectorStoreService:
    """
    RAG 기반 챗봇 서비스 (PostgreSQL + pgvector)
    - PDF 문서를 PostgreSQL 벡터 DB에 저장
    - 사용자 질문에 대해 문서 기반 답변 제공
    """

    def __init__(self, openai_api_key: str, database_url: str, collection_name: str = "document_embeddings"):
        """
        초기화 함수
        - OpenAI API 키 설정
        - 임베딩 모델 설정 (text-embedding-3-small)
        - PostgreSQL 연결 설정
        """
        self.openai_api_key = openai_api_key
        self.database_url = database_url
        self.collection_name = collection_name

        # 직접 환경변수 등록
        os.environ["OPENAI_API_KEY"] = openai_api_key

        # 임베딩 모델 설정 - 1536 차원
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        self.vectorstore: Optional[PGVector] = None  # 벡터 DB
        self.qa_chain: Optional[RetrievalQA] = None  # 질의응답 체인

    def create_vectorstore(self, documents, batch_size: int = 100):
        """
        벡터 스토어 생성 (배치 처리)
        - 문서들을 임베딩으로 변환
        - PostgreSQL에 저장
        - 대용량 문서 처리를 위해 배치 단위로 처리
        """
        print(f"총 {len(documents)}개의 문서 청크를 임베딩합니다...")

        # 기존 컬렉션 삭제 후 새로 생성
        if self.vectorstore:
            # 기존 벡터스토어가 있으면 삭제
            try:
                PGVector.from_existing_index(
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection=self.database_url,
                ).delete_collection()
                print("기존 벡터 스토어 삭제 완료")
            except:
                pass

        # 배치 단위로 나누어 처리
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            print(f"배치 {i // batch_size + 1}/{(len(documents) - 1) // batch_size + 1} 처리 중... ({len(batch)}개)")

            if i == 0:
                # 첫 번째 배치: 새 벡터스토어 생성
                self.vectorstore = PGVector.from_documents(
                    documents=batch,
                    embedding=self.embeddings,
                    collection_name=self.collection_name,
                    connection=self.database_url,
                    pre_delete_collection=True  # 기존 컬렉션 삭제
                )
            else:
                # 이후 배치: 기존 벡터스토어에 추가
                self.vectorstore.add_documents(batch)

        print("임베딩 완료!")
        return self.vectorstore

    def load_vectorstore(self):
        """
        기존 벡터 스토어 불러오기
        - 이미 저장된 벡터 DB가 있으면 재사용
        """
        try:
            self.vectorstore = PGVector(
                collection_name=self.collection_name,
                connection=self.database_url,
                embeddings=self.embeddings,
            )
            # 테스트 쿼리로 데이터 존재 확인
            result = self.vectorstore.similarity_search("test", k=1)
            if result:
                print(f"벡터 스토어 로드 완료: {len(result)}개 문서 확인")
                return self.vectorstore
            else:
                print("벡터 스토어가 비어있습니다.")
                return None
        except Exception as e:
            print(f"벡터 스토어 로드 실패: {e}")
            return None

    def create_qa_chain(self, model_name: str = "gpt-4o-mini"):
        """
        QA 체인 생성
        - LLM 모델 설정
        - 문서 검색 + 답변 생성 체인 구축
        """
        if not self.vectorstore:
            raise ValueError("벡터 스토어가 없습니다. create_vectorstore를 먼저 호출하세요.")

        # LLM 모델 설정
        llm = ChatOpenAI(
            model=model_name,
            temperature=0  # 일관된 답변을 위해 0으로 설정
        )

        # 친절한 한국어 프롬프트 추가
        from langchain_core.prompts import PromptTemplate

        prompt_template = """너는 친절한 한국어 비서야.
        아래 문서 내용을 참고해서 쉽고 자세하게 질문에 답해줘.
        답변할 때는 존댓말을 사용하고, 핵심 내용을 먼저 말한 후 자세한 설명을 해줘.

        문서 내용:
        {context}

        질문: {question}

        답변:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # QA 체인 생성 - 문서 검색 + LLM 답변 생성
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",  # 검색된 문서를 모두 프롬프트에 포함
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),  # 상위 3개 문서 검색
            return_source_documents=True,  # 참조 문서도 함께 반환
            chain_type_kwargs={"prompt": PROMPT}  # 커스텀 프롬프트 적용
        )
        return self.qa_chain

    def query(self, question: str) -> dict:
        """
        질문에 답변하기
        """
        if not self.qa_chain:
            raise ValueError("QA 체인이 없습니다. create_qa_chain을 먼저 호출하세요.")

        # 질문 실행 - 자동으로 문서 검색 + 답변 생성
        result = self.qa_chain.invoke({"query": question})

        return {
            "answer": result["result"],  # AI 답변
            "sources": [doc.page_content for doc in result["source_documents"]]  # 참조 문서
        }
