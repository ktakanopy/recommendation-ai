import json
from typing import Any, Dict

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from big_boss.domain.ports.services.llm import LLMPort
from big_boss.infrastructure.config.settings import BotSettings


class LangChainOpenAILLM(LLMPort):
    def __init__(self, settings: BotSettings):
        self.llm = ChatOpenAI(model="gpt-5-nano", api_key=settings.OPENAI_API_KEY, temperature=0.5)

    def extract_user_profile(self, text: str) -> Dict[str, Any]:
        sys = (
            "You are a smart data extractor. Extract the user's age (as integer) and occupation from their message. "
            "For age: Look for any number that could represent age (e.g., 'I'm 25', '30 years old', 'twenty-five'). "
            "For occupation: Match to ONE of these exact labels: other, academic/educator, artist, clerical/admin, "
            "college/grad student, customer service, doctor/health care, executive/managerial, farmer, homemaker, "
            "K-12 student, lawyer, programmer, retired, sales/marketing, scientist, self-employed, "
            "technician/engineer, tradesman/craftsman, unemployed, writer. "
            "Examples: 'software engineer' -> 'programmer', 'teacher' -> 'academic/educator', 'doctor' -> 'doctor/health care'. "
            "If you can't determine something, set it to null. "
            "Respond ONLY as valid JSON: {\"age\": int_or_null, \"occupation_label\": \"exact_label_or_null\"}"
        )
        msgs = [SystemMessage(content=sys), HumanMessage(content=f"User message: {text}")]
        resp = self.llm.invoke(msgs)
        content = getattr(resp, "content", "{}") or "{}"
        
        # Clean the response to extract JSON if wrapped in markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        try:
            data = json.loads(content)
            return {
                "age": data.get("age") if isinstance(data.get("age"), int) else None,
                "occupation_label": data.get("occupation_label") if isinstance(data.get("occupation_label"), str) else None
            }
        except Exception:
            return {"age": None, "occupation_label": None}

    def chat(self, system: str, user: str) -> str:
        msgs = [SystemMessage(content=system), HumanMessage(content=user)]
        resp = self.llm.invoke(msgs)
        return getattr(resp, "content", "") or ""

    def extract_liked_movie_ids(self, movies: Dict[str, Any], user_reply: str) -> Dict[str, Any]:
        sys = (
            "You are a helpful assistant that reads a list of movies and a user's reply, "
            "and extracts the IDs of the movies they liked. Only return IDs that exist in the provided list. "
            "Respond ONLY as valid JSON: {\"liked_ids\": [int, ...]}"
        )
        payload = json.dumps({"movies": movies, "reply": user_reply})
        msgs = [SystemMessage(content=sys), HumanMessage(content=payload)]
        resp = self.llm.invoke(msgs)
        content = getattr(resp, "content", "{}") or "{}"
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
        try:
            data = json.loads(content)
            liked = data.get("liked_ids")
            if isinstance(liked, list):
                return {"liked_ids": [int(x) for x in liked if isinstance(x, (int, str)) and str(x).isdigit()]}
        except Exception:
            pass
        return {"liked_ids": []}
