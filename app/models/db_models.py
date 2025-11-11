# app/models/db_models.py
from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.types import TypeDecorator, CHAR
from pgvector.sqlalchemy import Vector
from datetime import datetime, timezone
import uuid
from app.database import Base


class UUID(TypeDecorator):
    """SQLite와 PostgreSQL 모두 지원하는 UUID 타입"""
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == 'postgresql':
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == 'postgresql':
            return value
        else:
            if isinstance(value, uuid.UUID):
                return str(value)
            return value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        return uuid.UUID(value)


class User(Base):
    """사용자 모델"""
    __tablename__ = "users"

    user_id = Column(UUID(), primary_key=True, default=uuid.uuid4, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    password = Column(String(255), nullable=False)
    nickname = Column(String(100), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        """모델을 딕셔너리로 변환"""
        return {
            "user_id": str(self.user_id),
            "email": self.email,
            "nickname": self.nickname,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class DocumentEmbedding(Base):
    """문서 임베딩 모델 - pgvector를 사용한 벡터 검색"""
    __tablename__ = "document_embeddings"

    id = Column(UUID(), primary_key=True, default=uuid.uuid4, index=True)
    content = Column(Text, nullable=False)  # 문서 내용
    embedding = Column(Vector(1536), nullable=False)  # OpenAI text-embedding-3-small 차원
    doc_metadata = Column(JSONB, default=dict)  # 메타데이터 (파일명, 페이지 번호 등)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)

    def to_dict(self):
        """모델을 딕셔너리로 변환"""
        return {
            "id": str(self.id),
            "content": self.content,
            "doc_metadata": self.doc_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
