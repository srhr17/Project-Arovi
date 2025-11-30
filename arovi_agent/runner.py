import asyncio
import os

from google.adk.sessions import InMemorySessionService
from google.adk.runners import Runner
from google.genai import types

from .agents import arovi_root_agent, APP_NAME
from dotenv import load_dotenv
load_dotenv()


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

    # Compose a natural-language query for the root agent
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

    # The ADK Runner expects a list of Content messages (google.genai.types)
    content = types.Content(
        role="user",
        parts=[types.Part.from_text(user_message)],
    )

    print(f"\n=== Arovi input ===\n{user_message}\n")

    event_stream = runner.run(content=content, session_id=session_id)

    final_briefing = None
    async for event in event_stream:
        # You can print events for debugging or observability.
        # For the capstone you might want to show key events only.
        if event.content:
            # This will include the final briefing as well as intermediate text.
            print(event.content)

        # You could also inspect event.metadata to detect the last reply.

    # In a more advanced setup you would fetch session.state from the session_service
    # after the run is complete and print `briefing_revised` / `briefing_draft` from there.


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
