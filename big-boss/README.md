# Big Boss Bot

## Overview
A Telegram bot that onboards users, collects initial ratings, and shows recommendations by calling the neural_recommendation service. Built with a Clean Architecture layout and LangGraph for dialogue state management.

States:
- User Onboarding: ask demographics (gender, age, occupation) and create a user via API
- Ratings Onboarding: fetch onboarding movies, ask which ones the user liked, and create ratings
- Show Recommendations: fetch cold-start recommendations and display

## Setup
Create a .env with:
```
OPENAI_API_KEY=...
TELEGRAM_BOT_TOKEN=...
NR_BASE_URL=http://localhost:8000
```

Install dependencies with Poetry from the repo root:
```
cd bot
poetry install
```

## Run Telegram bot
```
poetry run python -m big_boss.presentation.telegram.runner
```

## Structure
```
big_boss/
  big_boss/
    domain/
      models/
      ports/
    applications/
      services/
      graphs/
    infrastructure/
      adapters/
      config/
    presentation/
      telegram/
        runner.py
```
