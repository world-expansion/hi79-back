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
        alternative_perspective: str,
        message_count: int
    ) -> str:
        """
        일기 저장

        Args:
            user_id: 사용자 ID
            diary_content: 일기 본문

        Returns:
            diary_id: 저장된 일기 ID
        """
        diary_id = str(uuid.uuid4())
        now = datetime.now()

        # 날짜 계산: 06:00 이전이면 전날로 기록
        if now.hour < 6:
            diary_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        else:
            diary_date = now.strftime("%Y-%m-%d")

        # 같은 날짜의 기존 일기 삭제
        self._delete_diary_by_date(user_id, diary_date)

        # 메타데이터 구성 (필수 필드만)
        doc_metadata = {
            "user_id": user_id,
            "diary_id": diary_id,
            "diary_date": diary_date,  # 일기 날짜 (06:00 기준)
            "created_at": now.isoformat(),  # 실제 생성 시각
            "alternative_perspective": alternative_perspective,
            "message_count": message_count
        }

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

    def _delete_diary_by_date(self, user_id: str, diary_date: str) -> None:
        """
        특정 날짜의 기존 일기 삭제 (내부 헬퍼 메서드)

        Args:
            user_id: 사용자 ID
            diary_date: 일기 날짜 (YYYY-MM-DD)
        """
        try:
            from sqlalchemy import create_engine, text

            engine = create_engine(self.database_url)
            with engine.connect() as conn:
                # collection_id 조회
                result = conn.execute(text("""
                    SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name LIMIT 1
                """), {"collection_name": self.collection_name})
                collection_row = result.fetchone()

                if not collection_row:
                    return  # collection이 없으면 삭제할 일기도 없음

                collection_id = collection_row[0]

                # 같은 날짜의 기존 일기 삭제
                delete_query = text("""
                    DELETE FROM langchain_pg_embedding
                    WHERE collection_id = :collection_id
                      AND cmetadata->>'user_id' = :user_id
                      AND cmetadata->>'diary_date' = :diary_date
                """)

                conn.execute(delete_query, {
                    "collection_id": collection_id,
                    "user_id": str(user_id),
                    "diary_date": diary_date
                })
                conn.commit()

        except Exception as e:
            print(f"기존 일기 삭제 실패 (무시하고 계속 진행): {e}")
            # 삭제 실패해도 새 일기는 저장되어야 하므로 예외를 발생시키지 않음

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
                # cmetadata의 user_id와 diary_date 필터링
                query = text("""
                    SELECT 
                        document,
                        cmetadata
                    FROM langchain_pg_embedding
                    WHERE collection_id = :collection_id
                      AND cmetadata->>'user_id' = :user_id
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
                    
                    diary_date_str = metadata.get("diary_date")
                    
                    if diary_date_str:
                        try:
                            # ISO 형식 문자열을 datetime으로 변환
                            diary_date = datetime.fromisoformat(diary_date_str.replace('Z', '+00:00'))
                            # 타임존 정보 제거 후 비교
                            diary_date_naive = diary_date.replace(tzinfo=None)
                            
                            # 최근 N일 이내인지 확인
                            if diary_date_naive >= cutoff_date:
                                weekly_diaries.append({
                                    "content": document_content,
                                    "metadata": metadata,
                                    "diary_date": diary_date_str
                                })
                        except (ValueError, AttributeError) as e:
                            # 날짜 파싱 실패 시 로그만 남기고 건너뛰기
                            print(f"날짜 파싱 실패: {diary_date_str}, 오류: {e}")
                            continue
                
                # 날짜순 정렬 (최신순)
                weekly_diaries.sort(
                    key=lambda x: x.get("diary_date", ""),
                    reverse=True
                )
                
                return weekly_diaries

        except Exception as e:
            print(f"일주일치 일기 조회 실패: {e}")
            import traceback
            traceback.print_exc()
            return []

    def get_diary_by_date(
        self,
        user_id: str,
        date: str
    ) -> Optional[Dict]:
        """
        특정 날짜의 일기 조회
        langchain_pg_embedding 테이블의 document 컬럼에서 직접 조회

        Args:
            user_id: 사용자 ID
            date: 조회할 날짜 (YYYY-MM-DD 형식)

        Returns:
            일기 데이터 (없으면 None)
        """
        try:
            from sqlalchemy import create_engine, text
            import json
            
            # 날짜 파싱 및 검증
            try:
                target_date = datetime.strptime(date, "%Y-%m-%d").date()
            except ValueError:
                raise ValueError(f"날짜 형식이 올바르지 않습니다. YYYY-MM-DD 형식으로 입력해주세요: {date}")
            
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
                    return None
                
                collection_id = collection_row[0]
                
                # langchain_pg_embedding 테이블에서 특정 날짜의 일기 조회
                # diary_date가 해당 날짜인 일기 찾기 (ISO 형식 문자열에서 날짜 부분만 비교)
                # diary_date가 '2025-11-12T10:30:00' 형식이므로 날짜 부분만 추출
                date_str = target_date.isoformat()  # '2025-11-12'
                query = text("""
                    SELECT 
                        document,
                        cmetadata
                    FROM langchain_pg_embedding
                    WHERE collection_id = :collection_id
                      AND cmetadata->>'user_id' = :user_id
                      AND (cmetadata->>'diary_date')::text LIKE :date_pattern
                    LIMIT 1
                """)
                
                # 날짜 패턴: '2025-11-12'로 시작하는 모든 시간 포함
                date_pattern = f"{date_str}%"
                
                result = conn.execute(query, {
                    "collection_id": collection_id,
                    "user_id": str(user_id),
                    "date_pattern": date_pattern
                })
                
                row = result.fetchone()
                
                if not row:
                    return None
                
                document_content = row[0]  # document 컬럼
                metadata_json = row[1]  # cmetadata 컬럼 (JSONB)
                
                # JSONB를 딕셔너리로 변환
                if isinstance(metadata_json, str):
                    metadata = json.loads(metadata_json)
                else:
                    metadata = metadata_json
                
                diary_date_str = metadata.get("diary_date", "")
                
                return {
                    "content": document_content,
                    "metadata": metadata,
                    "diary_date": diary_date_str
                }

        except ValueError as e:
            print(f"날짜 파싱 실패: {e}")
            raise
        except Exception as e:
            print(f"특정 날짜 일기 조회 실패: {e}")
            import traceback
            traceback.print_exc()
            return None


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
