# routers/admin.py
from fastapi import APIRouter, HTTPException, Depends
from app.schemas.scheduler import (
    SchedulerStatusResponse,
    DiaryCreationResponse,
)
from app.services.diary_scheduler import get_diary_scheduler
from app.routers.auth import get_current_user_id
from datetime import datetime
import traceback

router = APIRouter(prefix="/api/admin", tags=["Admin"])


@router.post("/diary/trigger", response_model=DiaryCreationResponse, summary="수동 일기 생성 트리거")
async def trigger_diary_creation(
    user_id: str = Depends(get_current_user_id)
):
    """
    수동으로 일기 생성 작업 트리거 (관리자/테스트용)

    - 새벽 6시를 기다리지 않고 즉시 실행
    - Redis의 모든 세션을 일기로 변환
    - 테스트 및 긴급 상황에 사용

    **주의:** 실행 중인 모든 세션이 종료되고 일기로 변환됩니다.
    """
    try:
        scheduler = get_diary_scheduler()

        # 비동기 작업 실행
        await scheduler.auto_create_diaries()

        return DiaryCreationResponse(
            success=True,
            message="일기 생성 작업이 완료되었습니다.",
            data={
                "triggered_by": user_id,
                "triggered_at": datetime.now().isoformat(),
                "status": "completed"
            }
        )

    except Exception as e:
        error_detail = f"일기 생성 실패: {str(e)}\n\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"일기 생성 실패: {str(e)}")


@router.get("/scheduler/status", response_model=SchedulerStatusResponse, summary="스케줄러 상태 확인")
async def get_scheduler_status(
    user_id: str = Depends(get_current_user_id)
):
    """
    자동 일기 생성 스케줄러 상태 확인

    - 실행 중인지 확인
    - 다음 실행 시간 확인
    - 등록된 작업 목록
    """
    try:
        scheduler = get_diary_scheduler()

        # 스케줄된 작업 정보
        jobs = scheduler.scheduler.get_jobs()

        if not jobs:
            return SchedulerStatusResponse(
                success=True,
                message="스케줄러가 실행 중이지 않습니다.",
                data={
                    "running": False,
                    "jobs": []
                }
            )

        job_info = []
        for job in jobs:
            job_info.append({
                "id": job.id,
                "name": job.name,
                "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
                "trigger": str(job.trigger)
            })

        return SchedulerStatusResponse(
            success=True,
            message="스케줄러가 정상 실행 중입니다.",
            data={
                "running": True,
                "job_count": len(jobs),
                "jobs": job_info
            }
        )

    except Exception as e:
        error_detail = f"상태 확인 실패: {str(e)}\n\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"상태 확인 실패: {str(e)}")


@router.get("/scheduler/next-run", summary="다음 실행 시간 확인")
async def get_next_run_time(
    user_id: str = Depends(get_current_user_id)
):
    """
    다음 자동 일기 생성 실행 시간 확인

    - 다음 실행까지 남은 시간 계산
    """
    try:
        scheduler = get_diary_scheduler()
        jobs = scheduler.scheduler.get_jobs()

        if not jobs:
            return {
                "success": True,
                "message": "예약된 작업이 없습니다.",
                "data": {
                    "next_run_time": None,
                    "remaining_seconds": None
                }
            }

        # 가장 빠른 실행 시간 찾기
        next_job = min(jobs, key=lambda j: j.next_run_time if j.next_run_time else datetime.max)

        if not next_job.next_run_time:
            return {
                "success": True,
                "message": "다음 실행 시간이 설정되지 않았습니다.",
                "data": {
                    "next_run_time": None,
                    "remaining_seconds": None
                }
            }

        # 남은 시간 계산
        now = datetime.now(next_job.next_run_time.tzinfo)
        remaining = (next_job.next_run_time - now).total_seconds()

        return {
            "success": True,
            "message": "다음 실행 시간 조회 성공",
            "data": {
                "job_name": next_job.name,
                "next_run_time": next_job.next_run_time.isoformat(),
                "remaining_seconds": int(remaining),
                "remaining_minutes": int(remaining / 60),
                "remaining_hours": round(remaining / 3600, 2)
            }
        }

    except Exception as e:
        error_detail = f"조회 실패: {str(e)}\n\n{traceback.format_exc()}"
        print(error_detail)
        raise HTTPException(status_code=500, detail=f"조회 실패: {str(e)}")
