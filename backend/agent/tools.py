# backend/agent/tools.py (신규 파일)

import os, json, datetime as dt
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic

# 필요한 모델 및 헬퍼 함수 임포트
import models
from routers.gcal import build_gcal_service
from routers.search import google_search_cse
from utils.image import fetch_and_resize
from .mcp_loader import load_mcp_tools

# Pydantic 스키마 (기존 __init__.py에서 이동)
class CreateEventArgs(BaseModel):
    title: str  = Field(..., description="일정 제목")
    start: str  = Field(..., description="ISO-8601 시작")
    end:   str  = Field(..., description="ISO-8601 종료")
class DeleteEventArgs(BaseModel):
    event_id: str
class WebSearchArgs(BaseModel):
    query: str
    k: int = 5
class GenImgArgs(BaseModel):
    prompt: str = Field(..., description="DALL-E 프롬프트")
class RecArgs(BaseModel):
    types: str
    limit: int = 5
class ExtractTitleArgs(BaseModel):
    text_to_process: str = Field(..., description="The raw text from a previous search/recommendation step.")

# 도구 세트 생성 함수 (기존 __init__.py에서 이동)
def make_toolset(db, user, tz, llm_instance: ChatAnthropic):

    @tool(args_schema=CreateEventArgs, return_direct=True)
    def create_event(title: str, start: str, end: str) -> str:
        """Google Calendar 일정 생성. 사용자가 일정, 미팅, 약속 등을 잡아달라고 할 때 항상 사용하세요.
        
        title: 일정 제목 (예: "팀 회의", "점심 약속")
        start: ISO-8601 시작 시간 (예: "2025-05-26T13:00:00+02:00")
        end: ISO-8601 종료 시간 (예: "2025-05-26T14:00:00+02:00")
        
        일정 생성은 항상 미래 시간에만 가능합니다.
        """
        print("---------------------------------------")
        print("create_event 호출됨!")
        print(f"매개변수: title={title}, start={start}, end={end}")
        print("---------------------------------------")
        try:
            # 1) ISO → datetime
            try:
                dt_start = dt.datetime.fromisoformat(start)
                dt_end = dt.datetime.fromisoformat(end)
            except ValueError as e:
                print(f"ISO 파싱 실패: {start}, {end}, 오류: {e}")
                return f"❗ 날짜 형식이 올바르지 않습니다: {e}"

            # 2) 타임존 처리
            if dt_start.tzinfo is None:
                dt_start = dt_start.replace(tzinfo=tz)
            if dt_end.tzinfo is None:
                dt_end = dt_end.replace(tzinfo=tz)
            
            # 3) 현재 시간
            now = dt.datetime.now(tz)
            print(f"시간 비교: 시작={dt_start}, 현재={now}")
            
            # 4) 미래 일정 확인 (10분 이내는 허용)
            if dt_start < now - dt.timedelta(minutes=10):
                return f"❗ 과거 시간({dt_start.strftime('%Y-%m-%d %H:%M')})에는 일정을 추가할 수 없습니다. 현재 시간은 {now.strftime('%Y-%m-%d %H:%M')}입니다."
            
            # 5) Google Calendar API 호출
            svc = build_gcal_service(db, user.id)
            print(f"일정 생성 시도: {title}, {dt_start} ~ {dt_end}")
            ev = svc.events().insert(
                calendarId="primary",
                body={
                    "summary": title,
                    "start": {"dateTime": dt_start.isoformat(), "timeZone": str(tz)},
                    "end": {"dateTime": dt_end.isoformat(), "timeZone": str(tz)},
                },
            ).execute()

            result = f"✅ 일정 생성 완료 → {dt_start.strftime('%Y-%m-%d %H:%M')} ~ {dt_end.strftime('%H:%M')} {ev.get('htmlLink')}"
            print(result)
            return result
        except Exception as e:
            print(f"일정 생성 오류: {e}")
            return f"❗ 일정 생성 중 오류가 발생했습니다: {str(e)}"

    @tool(args_schema=DeleteEventArgs, return_direct=True)
    def delete_event(event_id: str) -> str:
        """event_id 로 Google Calendar 이벤트를 삭제한다."""
        svc = build_gcal_service(db, user.id)
        svc.events().delete(calendarId="primary", eventId=event_id).execute()
        return "🗑️ 일정이 삭제되었습니다."

    @tool(args_schema=WebSearchArgs)
    def web_search(query: str, k: int = 5) -> str:
        """Google CSE 로 웹을 검색하고 상위 k개 링크를 돌려준다."""
        items = google_search_cse(query=query, num=k, date_restrict="m6", sort="date")
        return "\n".join(f"{it['title']} – {it['link']}" for it in items) or "No results"

    @tool(args_schema=GenImgArgs, return_direct=True)
    def generate_image(prompt: str) -> str:
        """이미지 생성 기능은 현재 비활성화되어 있습니다."""
        return json.dumps({"error": "이미지 생성 기능은 현재 지원되지 않습니다."})

    @tool(args_schema=RecArgs, return_direct=True)
    def fetch_recommendations(types: str, limit: int = 5) -> str:
        """
        ONLY USE THIS TOOL when the user EXPLICITLY asks for content recommendations or suggestions.
        
        APPROPRIATE USES:
        - User asks "뭐 볼까?" (What should I watch?)
        - User says "영화 추천해줘" (Recommend me a movie)
        - User asks for options or suggestions for content to consume
        
        DO NOT USE FOR:
        - Technical questions like "React란 무엇인가?"
        - Factual information queries like "TypeScript의 장점은?"
        - General knowledge or explanations
        
        This tool returns personalized content recommendation cards as JSON.
        """
        from routers.recommend import get_recommendations
        recs = get_recommendations(
            types=types, limit=limit, db=db, current_user=user, tz=tz, user_query=""
        )
        return json.dumps({"cards": recs}, ensure_ascii=False)

    @tool(args_schema=ExtractTitleArgs)
    def extract_best_title(text_to_process: str) -> str:
        """
        Processes raw text from search or recommendation results to extract the single most relevant item title for a calendar event.
        Use this to clean up the output of a search before creating a calendar event.
        """
        
        prompt = ChatPromptTemplate.from_template(
            "From the following search results, extract the single most relevant movie or event title. "
            "Return ONLY the title itself, with no extra words, explanations, or quotes.\n\n"
            "SEARCH RESULTS:\n{text}\n\n"
            "TITLE:"
        )

        chain = prompt | llm_instance
        
        try:
            response = chain.invoke({"text": text_to_process})
            # LLM 응답에서 불필요한 따옴표 등을 제거
            extracted_title = response.content.strip().strip('"')
            print(f"      - Extracted Title: '{extracted_title}'")
            return extracted_title
        except Exception as e:
            print(f"      - Title extraction failed: {e}")
            # 실패 시 기본값 반환
            return "선택된 항목"

    base_tools = [
        create_event,
        delete_event,
        web_search,
        generate_image,
        fetch_recommendations,
        extract_best_title,
    ]

    # MCP 로드 (실패해도 base_tools 그대로)
    mcp_tools = load_mcp_tools(host=os.getenv("MCP_HOST","mcp-weather"),
                               port=int(os.getenv("MCP_PORT","7001")))
    return base_tools + mcp_tools