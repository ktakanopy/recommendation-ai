import asyncio
import re
import logging
import json
from typing import Any, Dict, List, Optional, Tuple

import requests
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from big_boss.infrastructure.config.settings import BotSettings
from big_boss.domain.ports.services.llm import LLMPort


logger = logging.getLogger(__name__)


class BotState(BaseModel):
    user_id: Optional[int] = None
    stage: str = "onboarding"
    buffer: Dict[str, Any] = {}
    last_output: str = ""


class OnboardingGraph:
    def __init__(self, settings: BotSettings, llm_port: Optional[LLMPort] = None):
        self.settings = settings
        self.base_url = settings.NR_BASE_URL.rstrip("/")
        self.graph = self._build_graph()
        self.llm = llm_port
        self.sessions: Dict[str, BotState] = {}

    async def _req(self, method: str, path: str, *, json: Any | None = None, params: Dict[str, Any] | None = None):
        url = f"{self.base_url}{path}"
        logger.info("http %s %s json=%s params=%s", method, url, bool(json), params)
        try:
            resp = await asyncio.to_thread(requests.request, method, url, json=json, params=params, timeout=20)
            logger.info("http done %s %s status=%s", method, url, resp.status_code)
            return resp
        except Exception:
            logger.exception("http error %s %s", method, url)
            raise


    def _coerce_state(self, value: Any) -> BotState:
        if isinstance(value, BotState):
            return value
        if isinstance(value, dict):
            return BotState(
                user_id=value.get("user_id"),
                stage=value.get("stage", "onboarding"),
                buffer=value.get("buffer", {}),
                last_output=value.get("last_output", ""),
            )
        raise TypeError("Invalid graph result")


    async def _create_user(self, state: BotState) -> BotState:
        data = state.buffer
        payload = {
            "username": data.get("username", f"user_{asyncio.get_event_loop().time():.0f}"),
            "email": data.get("email", f"user{asyncio.get_event_loop().time():.0f}@example.com"),
            "password": data.get("password", "changeme123"),
            "age": data.get("age"),
            "gender": data.get("gender", "M"),
            "occupation": data.get("occupation"),
        }
        logger.info("create_user payload=%s", {k: payload[k] for k in ["age", "gender", "occupation", "username"]})
        r = await self._req("POST", "/users/", json=payload)
        r.raise_for_status()
        user = r.json()
        state.user_id = user["id"]
        state.stage = "ratings_onboarding"
        state.last_output = "User created. Getting onboarding movies..."
        logger.info("user_created user_id=%s", state.user_id)
        return state

    async def _get_onboarding_movies(self, state: BotState) -> BotState:
        r = await self._req("GET", "/recommendations/onboarding-movies", json={"num_movies": 5})
        if r.status_code >= 400:
            r = await self._req("GET", "/recommendations/onboarding-movies", params={"num_movies": 5})
        r.raise_for_status()
        res = r.json()
        state.buffer["onboarding_movies"] = res.get("recommendations", res)
        if self.llm is not None:
            listing = json.dumps(state.buffer["onboarding_movies"]) if not isinstance(state.buffer["onboarding_movies"], str) else state.buffer["onboarding_movies"]
            prompt = self.llm.chat(
                system="You are a helpful assistant. Show the user a friendly list of movies with their IDs, then ask them to reply with the IDs they liked.",
                user=listing,
            )
            state.last_output = prompt.strip() or "Please reply with the movie ids you liked, separated by commas."
        else:
            state.last_output = "Please reply with the movie ids you liked, separated by commas."
        logger.info("onboarding_movies_loaded count=%s", sum(len(v) for v in state.buffer["onboarding_movies"].values()) if isinstance(state.buffer["onboarding_movies"], dict) else len(state.buffer["onboarding_movies"]))
        return state

    async def _create_ratings(self, state: BotState) -> BotState:
        liked: List[int] = state.buffer.get("liked_movie_ids", [])
        payload = [{"user_id": state.user_id, "movie_id": mid, "rating": 5.0} for mid in liked]
        if payload:
            logger.info("create_ratings user_id=%s count=%s", state.user_id, len(payload))
            r = await self._req("POST", "/ratings/", json=payload)
            r.raise_for_status()
        state.stage = "recommend"
        state.last_output = "Ratings saved. Fetching recommendations..."
        return state

    async def _recommend(self, state: BotState) -> BotState:
        payload = {"user_id": state.user_id, "num_recommendations": 10}
        logger.info("recommend user_id=%s", state.user_id)
        r = await self._req("POST", "/recommendations/cold-start", json=payload)
        r.raise_for_status()
        res = r.json()
        recs = res.get("recommendations", [])
        lines = [f"{i+1}. {x['title']} (id={x['movie_id']}) score={x.get('similarity_score', 0):.3f}" for i, x in enumerate(recs)]
        state.last_output = "\n".join(lines) if lines else "No recommendations available."
        logger.info("recommendations user_id=%s count=%s", state.user_id, len(recs))
        return state

    def _build_graph(self):
        g = StateGraph(BotState)

        async def onboarding(state: BotState) -> BotState:
            logger.info("node:onboarding stage=%s has_age=%s has_occ=%s", state.stage, "age" in state.buffer, "occupation" in state.buffer)
            if state.user_id is not None:
                return state
            if "age" not in state.buffer:
                if self.llm is not None:
                    prompt = self.llm.chat(
                        system="You are a friendly onboarding assistant. Ask the user for their age in a short, natural way.",
                        user="User just started the conversation.",
                    )
                    state.last_output = prompt.strip() or "Hi! What is your age?"
                else:
                    state.last_output = "Hi! What is your age?"
                return state
            if "occupation" not in state.buffer:
                if self.llm is not None:
                    prompt = self.llm.chat(
                        system=(
                            "You are a friendly onboarding assistant. Ask for the user's occupation in a brief, human way. "
                            "Make clear they can just write it naturally, like 'programmer' or 'teacher'."
                        ),
                        user="The user already provided their age.",
                    )
                    state.last_output = prompt.strip() or "Great. What is your occupation? You can write it naturally, for example: programmer, artist, lawyer, engineer, teacher."
                else:
                    state.last_output = "Great. What is your occupation? You can write it naturally, for example: programmer, artist, lawyer, engineer, teacher."
                return state
            return await self._create_user(state)

        async def ratings_onboarding(state: BotState) -> BotState:
            logger.info("node:ratings_onboarding stage=%s has_movies=%s has_liked=%s", state.stage, "onboarding_movies" in state.buffer, "liked_movie_ids" in state.buffer)
            if "onboarding_movies" not in state.buffer:
                return await self._get_onboarding_movies(state)
            if "liked_movie_ids" in state.buffer:
                return await self._create_ratings(state)
            if self.llm is not None and "user_reply" in state.buffer:
                try:
                    extracted = self.llm.extract_liked_movie_ids(state.buffer.get("onboarding_movies", {}), state.buffer.get("user_reply", ""))
                    liked_ids = extracted.get("liked_ids", [])
                    if liked_ids:
                        state.buffer["liked_movie_ids"] = liked_ids
                        return await self._create_ratings(state)
                except Exception:
                    logger.exception("llm extract_liked_movie_ids failed")
            state.last_output = state.last_output or "Please reply with the movie ids you liked, separated by commas."
            return state

        async def recommend(state: BotState) -> BotState:
            logger.info("node:recommend stage=%s", state.stage)
            return await self._recommend(state)

        g.add_node("onboarding", onboarding)
        g.add_node("ratings_onboarding", ratings_onboarding)
        g.add_node("recommend", recommend)

        def route(state: BotState) -> str:
            return state.stage

        g.set_entry_point("onboarding")
        g.add_edge("onboarding", "ratings_onboarding")
        g.add_edge("ratings_onboarding", "recommend")
        g.add_edge("recommend", END)
        return g.compile()

    async def handle_message(self, user_id: str, message: str) -> str:
        state = self.sessions.get(user_id) or BotState()
        text = message.strip()
        logger.info("handle_message user_id=%s text=%s", user_id, text)
        if text and self.llm is not None:
            try:
                data = await asyncio.to_thread(self.llm.extract_user_profile, text)
                logger.info("extracted age=%s occupation_label=%s", data.get("age"), data.get("occupation_label"))
                if data.get("age") is not None:
                    try:
                        age = int(data["age"]) if isinstance(data["age"], int) else int(str(data["age"]))
                        if 1 <= age <= 120:
                            state.buffer["age"] = age
                    except (ValueError, TypeError):
                        logger.warning("invalid age value=%s", data.get("age"))
                if data.get("occupation_label"):
                    mapped = self._map_occupation(str(data["occupation_label"]))
                    if mapped is not None:
                        code, label = mapped
                        state.buffer["occupation"] = code
                        state.last_output = f"Occupation detected as {label} (code {code})."
            except Exception:
                logger.exception("llm extract_user_profile failed")
                raise Exception("Failed to extract user profile")
        raw = await self.graph.ainvoke(state)
        result_state = self._coerce_state(raw)
        self.sessions[user_id] = result_state
        logger.info("after_graph stage=%s has_age=%s has_occ=%s", result_state.stage, "age" in result_state.buffer, "occupation" in result_state.buffer)
        if result_state.stage == "onboarding" and "age" in result_state.buffer and "occupation" not in result_state.buffer and not result_state.last_output:
            result_state.last_output = "Great. What is your occupation? You can write it naturally, for example: programmer, artist, lawyer, engineer, teacher."
        return result_state.last_output
