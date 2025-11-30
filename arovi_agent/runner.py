import asyncio
import os

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from .agents import arovi_root_agent, APP_NAME
from dotenv import load_dotenv
load_dotenv()

import os

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment. Set it in .env.")



USER_ID = "demo_user"
SESSION_ID = "arovi_session_1"


async def create_runner() -> tuple[Runner, str]:
    """
    Creates an InMemorySessionService + Runner for local testing.

    For production / Agent Engine, you'd typically:
      - Use VertexAiSessionService
      - Use VertexAiMemoryBankService for long-term memory
    """
    session_service = InMemorySessionService()

    # Create or reuse session
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )

    runner = Runner(
        agent=arovi_root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )

    return runner, SESSION_ID


async def run_arovi_once(
    city: str,
    state: str | None = None,
    country: str = "United States",
    date_str: str | None = None,
):
    runner, session_id = await create_runner()

    # Compose user message
    pieces = [f"Generate a calm, public-health daily briefing for {city}"]
    if state:
        pieces.append(f"in the state/region {state}")
    if country:
        pieces.append(f"in {country}")
    if date_str:
        pieces.append(f"for the date {date_str}")
    else:
        pieces.append("for today")

    user_message = " ".join(pieces)

    from google.genai import types

    content = types.Content(
        role="user",
        parts=[types.Part.from_text(text=user_message)],
    )

    print(f"\n=== Arovi input ===\n{user_message}\n")

    event_stream = runner.run(
        user_id=USER_ID,
        session_id=session_id,
        new_message=content,
    )

    # Stream events (optional – useful during dev)
    for event in event_stream:
        if event.content:
            print(event.content)

    # 🔥 NEW: after runner finishes, fetch session.state and print final briefing
    session = await runner.session_service.get_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=session_id,
    )
    state = session.state or {}

    final_briefing = state.get("briefing_revised") or state.get("briefing_draft")

    print("\n\n=== FINAL AROVI BRIEFING ===\n")
    if final_briefing:
        print(final_briefing)
    else:
        print(
            "No final briefing found in state. "
            "Check that combiner_agent and redraft_agent are writing to "
            "`briefing_draft` / `briefing_revised`."
        )



def main():
    # Simple manual test: Chicago today
    asyncio.run(
        run_arovi_once(
            city="Chicago",
            state="Illinois",
            country="United States",
            date_str=None,
        )
    )


if __name__ == "__main__":
    main()
