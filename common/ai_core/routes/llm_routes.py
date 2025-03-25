from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from ai_core.config import configs
from ai_core.data_models.inference import (
    GenerateTextRequest,
    GenerateResponseRequest,
    StatelessChatRequest,
    BrainstormRequest,
    TokenCountResponse,
    TextBatchInput
)
from ai_core.services.llm_service import  llm_instance

router = APIRouter()



@router.post("/generate_text")
def generate_text(request: GenerateTextRequest):
    try:
        result = llm_instance.generate_text(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            skip_special_tokens=request.skip_special_tokens,
            **(request.kwargs or {})
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Oops! An error occurred in generate_text: {str(e)}")

@router.post("/generate_response")
def generate_response(request: GenerateResponseRequest):
    try:
        result = llm_instance.generate_response(
            prompt=request.prompt,
            system_prompt=request.system_prompt,
            tools_schema=request.tools_schema,
            documents=request.documents,
            create_chat_session=request.create_chat_session,
            chat_history=request.chat_history,
            max_new_tokens=request.max_new_tokens,
            skip_special_tokens=request.skip_special_tokens,
            **(request.kwargs or {})
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alas! An error occurred in generate_response: {str(e)}")

@router.post("/stateless_chat")
def stateless_chat(request: StatelessChatRequest):
    try:
        result = llm_instance.stateless_chat(
            prompt=request.prompt,
            chat_history=request.chat_history,
            **(request.kwargs or {})
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Whoops! An error occurred in stateless_chat: {str(e)}")

@router.post("/brainstorm")
def brainstorm(request: BrainstormRequest):
    try:
        result = llm_instance.brainstrom(
            messages=request.messages,
            role=request.role or "",
            role_play_configs=request.role_play_configs,
            ai_assisted_turns=request.ai_assisted_turns,
            iam=request.iam or None,
            **(request.kwargs or {})
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yikes! An error occurred in brainstorm: {str(e)}")

@router.get("/healthcheck")
def healthcheck():
    try:
        return llm_instance.healthcheck()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Healthcheck error: {str(e)}")

@router.get("/get_templates")
def get_templates():
    try:
        return llm_instance.get_templates()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Template retrieval error: {str(e)}")

@router.get("/examples")
def get_examples():
    """
    Returns example payloads for each endpoint as a handy guide.
    """
    examples = {
        "generate_text": GenerateTextRequest.Config.json_schema_extra["example"],
        "generate_response": GenerateResponseRequest.Config.json_schema_extra["example"],
        "stateless_chat": StatelessChatRequest.Config.json_schema_extra["example"],
        "brainstorm": BrainstormRequest.Config.json_schema_extra["example"]
    }
    return examples


@router.post("/get-token-count", response_model=TokenCountResponse)
async def get_token_count(request: TextBatchInput):
    """Get the token count for the given text."""
    try:
        return {"token_counts": [llm_instance.get_token_count(text) for text in request.texts]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


