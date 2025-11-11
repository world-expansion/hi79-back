# schemas/chat.py
from pydantic import BaseModel, Field
from typing import Optional, List

class SessionCreateRequest(BaseModel):
    """세션 생성 요청"""
    pass  # user_id는 JWT에서 추출

class SessionCreateResponse(BaseModel):
    """세션 생성 응답"""
    success: bool
    message: str
    data: dict

class ChatMessageRequest(BaseModel):
    """채팅 메시지 요청"""
    session_id: str = Field(..., description="세션 ID")
    message: str = Field(..., min_length=1, max_length=2000, description="사용자 메시지")

class ChatMessageResponse(BaseModel):
    """채팅 메시지 응답"""
    success: bool
    message: str
    data: dict  # answer, similar_diaries 등

class SessionEndRequest(BaseModel):
    """세션 종료 요청"""
    session_id: str = Field(..., description="종료할 세션 ID")

class SessionEndResponse(BaseModel):
    """세션 종료 응답"""
    success: bool
    message: str
    data: dict  # diary_id, diary_content 등

class DiaryEntry(BaseModel):
    """일기 항목"""
    content: str = Field(..., description="일기 내용")
    metadata: dict = Field(..., description="일기 메타데이터")
    created_at: str = Field(..., description="생성 일시 (ISO 형식)")

class WeeklyDiariesData(BaseModel):
    """일주일치 일기 조회 데이터"""
    diaries: List[DiaryEntry] = Field(..., description="일기 리스트 (날짜순 정렬, 최신순)")
    count: int = Field(..., description="일기 개수")
    days: int = Field(..., description="조회한 일수")

class WeeklyDiariesResponse(BaseModel):
    """일주일치 일기 조회 응답"""
    success: bool = Field(..., description="성공 여부")
    message: str = Field(..., description="응답 메시지")
    data: WeeklyDiariesData = Field(..., description="일기 조회 데이터")
