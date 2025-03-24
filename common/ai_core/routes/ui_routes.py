from fastapi import APIRouter
from fastapi.responses import HTMLResponse
from pathlib import Path

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent

@router.get("/ui/chat", response_class=HTMLResponse)
async def chat_ui():
    """Serve Chat UI"""
    html_content = (BASE_DIR / "../templates/chat.html").read_text()
    return HTMLResponse(content=html_content)
