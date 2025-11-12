# services/diary_service.py
from typing import List, Dict, Optional
from datetime import datetime, timedelta
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from sqlalchemy.orm import Session
from app.config import get_settings
import uuid

class DiaryService:
    """
    일기 관리 서비스 (PostgreSQL + pgvector)
    - 일기 저장 (벡터 임베딩 포함)
    - 유사 일기 검색 (RAG)
    - user_id별 네임스페이스 분리
    """

    def __init__(self):
        settings = get_settings()
        self.openai_api_key = settings.openai_api_key
        self.database_url = settings.database_url

        # 임베딩 모델 (text-embedding-3-small, 1536 차원)
        self.embeddings = OpenAIEmbeddings(model='text-embedding-3-small')

        # 일기 전용 컬렉션
        self.collection_name = "user_diaries"

    def save_diary(
        self,
        user_id: str,
        diary_content: str,
        metadata: Optional[Dict] = None
    ) -> str:
        """
        일기 저장

        Args:
            user_id: 사용자 ID
            diary_content: 일기 본문
            metadata: 메타데이터 (선택) - 날짜, 감정, CBT 요소 등

        Returns:
            diary_id: 저장된 일기 ID
        """
        diary_id = str(uuid.uuid4())

        # 메타데이터 구성
        doc_metadata = metadata or {}
        doc_metadata.update({
            "user_id": user_id,
            "diary_id": diary_id,
            "created_at": datetime.now().isoformat(),
            "type": "diary"  # 구분자
        })

        # Document 객체 생성
        document = Document(
            page_content=diary_content,
            metadata=doc_metadata
        )

        # PGVector에 저장
        vectorstore = PGVector.from_documents(
            documents=[document],
            embedding=self.embeddings,
            collection_name=self.collection_name,
            connection=self.database_url,
            pre_delete_collection=False  # 기존 일기 유지
        )

        return diary_id

    def search_similar_diaries(
        self,
        user_id: str,
        query: str,
        k: int = 3
    ) -> List[Dict]:
        """
        유사 일기 검색 (RAG 리트리벌)

        Args:
            user_id: 사용자 ID (네임스페이스 필터)
            query: 검색 쿼리 (사용자 메시지)
            k: 반환할 일기 개수 (기본 3개)

        Returns:
            유사 일기 리스트 (content + metadata)
        """
        try:
            # PGVector 인스턴스 생성
            vectorstore = PGVector(
                collection_name=self.collection_name,
                connection=self.database_url,
                embeddings=self.embeddings
            )

            # 유사도 검색 (user_id 필터링)
            # PGVector는 metadata 필터를 지원합니다
            results = vectorstore.similarity_search(
                query,
                k=k,
                filter={"user_id": user_id}  # user_id로 필터링
            )

            # 결과 포맷팅
            diaries = []
            for doc in results:
                diaries.append({
                    "content": doc.page_content,
                    "metadata": doc.metadata
                })

            return diaries

        except Exception as e:
            print(f"일기 검색 실패: {e}")
            return []

    def get_user_diary_count(self, user_id: str) -> int:
        """
        사용자의 총 일기 개수 조회

        Args:
            user_id: 사용자 ID

        Returns:
            일기 개수
        """
        try:
            vectorstore = PGVector(
                collection_name=self.collection_name,
                connection=self.database_url,
                embeddings=self.embeddings
            )

            # user_id로 필터링하여 모든 일기 검색
            results = vectorstore.similarity_search(
                "",  # 빈 쿼리
                k=1000,  # 충분히 큰 수
                filter={"user_id": user_id}
            )

            return len(results)

        except:
            return 0

    def get_weekly_diaries(
        self,
        user_id: str,
        days: int = 7
    ) -> List[Dict]:
        """
        일주일치 일기 조회 (최근 N일)
        langchain_pg_embedding 테이블의 document 컬럼에서 직접 조회

        Args:
            user_id: 사용자 ID
            days: 조회할 일수 (기본 7일)

        Returns:
            일기 리스트 (content + metadata), 날짜순 정렬 (최신순)
        """
        try:
            from sqlalchemy import create_engine, text
            import json
            
            # 날짜 범위 계산 (현재 시간 기준)
            now = datetime.now()
            cutoff_date = now - timedelta(days=days)
            
            # collection_id 가져오기
            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                # collection_id 조회
                result = conn.execute(text("""
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name LIMIT 1
                """), {"collection_name": self.collection_name})
                collection_row = result.fetchone()
                
                if not collection_row:
                    print(f"Collection '{self.collection_name}'를 찾을 수 없습니다.")
                    return []
                
                collection_id = collection_row[0]
                
                # langchain_pg_embedding 테이블에서 직접 조회
                # cmetadata의 user_id와 created_at으로 필터링
                query = text("""
                    SELECT 
                        document,
                        cmetadata
                    FROM langchain_pg_embedding
                    WHERE collection_id = :collection_id
                      AND cmetadata->>'user_id' = :user_id
                      AND cmetadata->>'type' = 'diary'
                """)
                
                results = conn.execute(query, {
                    "collection_id": collection_id,
                    "user_id": str(user_id)
                })
                
                # 날짜 필터링 및 포맷팅
                weekly_diaries = []
                for row in results:
                    document_content = row[0]  # document 컬럼
                    metadata_json = row[1]  # cmetadata 컬럼 (JSONB)
                    
                    # JSONB를 딕셔너리로 변환
                    if isinstance(metadata_json, str):
                        metadata = json.loads(metadata_json)
                    else:
                        metadata = metadata_json
                    
                    created_at_str = metadata.get("created_at")
                    
                    if created_at_str:
                        try:
                            # ISO 형식 문자열을 datetime으로 변환
                            created_at = datetime.fromisoformat(created_at_str.replace('Z', '+00:00'))
                            # 타임존 정보 제거 후 비교
                            created_at_naive = created_at.replace(tzinfo=None)
                            
                            # 최근 N일 이내인지 확인
                            if created_at_naive >= cutoff_date:
                                weekly_diaries.append({
                                    "content": document_content,
                                    "metadata": metadata,
                                    "created_at": created_at_str
                                })
                        except (ValueError, AttributeError) as e:
                            # 날짜 파싱 실패 시 로그만 남기고 건너뛰기
                            print(f"날짜 파싱 실패: {created_at_str}, 오류: {e}")
                            continue
                
                # 날짜순 정렬 (최신순)
                weekly_diaries.sort(
                    key=lambda x: x.get("created_at", ""),
                    reverse=True
                )
                
                return weekly_diaries

        except Exception as e:
            print(f"일주일치 일기 조회 실패: {e}")
            import traceback
            traceback.print_exc()
            return []


# 전역 일기 서비스 인스턴스
_diary_service: Optional[DiaryService] = None

def get_diary_service() -> DiaryService:
    """
    일기 서비스 의존성 주입
    """
    global _diary_service
    if _diary_service is None:
        _diary_service = DiaryService()
    return _diary_service
