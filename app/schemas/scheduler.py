# schemas/scheduler.py
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime


class SchedulerJobInfo(BaseModel):
    """스케줄러 작업 정보"""
    id: str = Field(..., description="작업 ID")
    name: str = Field(..., description="작업 이름")
    next_run_time: Optional[str] = Field(None, description="다음 실행 시간 (ISO 8601)")
    trigger: str = Field(..., description="트리거 정보")


class SchedulerStatusResponse(BaseModel):
    """스케줄러 상태 응답"""
    success: bool
    message: str
    data: dict


class DiaryCreationResult(BaseModel):
    """일기 생성 결과"""
    session_id: str
    user_id: str
    diary_id: Optional[str] = None
    status: str  # "success" | "failed" | "skipped"
    message: Optional[str] = None


class DiaryCreationResponse(BaseModel):
    """일기 생성 응답"""
    success: bool
    message: str
    data: dict
