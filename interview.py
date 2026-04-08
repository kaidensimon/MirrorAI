import asyncio
import json
import logging
import time
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io, TurnHandlingOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env")

with open("interview_test.json") as f:
    QUESTIONS = json.load(f)["questions"]


class InterviewAgent(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly interviewer doing an interviewr.
            You will be given a question to ask. There will be a timer for each question. During that time, your job is to continue asking
            follow up questions based on the participants answers. Do not try to ask a different question from the one you were given.
            When it's time to ask a new question, you will be given it. WHEN YOU ARE GIVEN A QUESTION ONLY ASK THAT QUESTION IMMEDIETELY CEASE ALL ATTEMPTS AT FOLLOWING UP ON A PAST ANSWER"""
        )


server = AgentServer()


@server.rtc_session(agent_name="interview-agent")
async def interview_agent(ctx: agents.JobContext):
    conversation_log: list[tuple[str, str]] = []
    last_user_speech_at: list[float] = [0.0]
    last_agent_speech_at: list[float] = [0.0]

    session = AgentSession(
        stt="deepgram/nova-3:multi",
        llm="openai/gpt-4.1-mini",
        tts="deepgram/aura-2-athena-en",
        vad=silero.VAD.load(),
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            interruption={"enabled": False},
            
        ),

    )

    @session.on("agent_speech_committed")
    def _on_agent_speech(ev):
        text = getattr(ev, "agent_transcript", None) or getattr(ev, "transcript", None) or str(ev)
        conversation_log.append(("agent", text))
        last_agent_speech_at[0] = time.monotonic()

    @session.on("user_speech_committed")
    def _on_user_speech(ev):
        text = getattr(ev, "user_transcript", None) or getattr(ev, "transcript", None) or str(ev)
        conversation_log.append(("user", text))
        last_user_speech_at[0] = time.monotonic()

    await session.start(
        room=ctx.room,
        agent=InterviewAgent(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await session.say("sup nigga thx 4 coming to interview")

    dialogue = []

    for i, question in enumerate(QUESTIONS):
        q_text = question["question"]
        time_limit = question["time_limit_seconds"]

        # Snapshot log before asking so we can slice the window later
        log_start = len(conversation_log)

        # Ask the question
        await session.say(q_text)

        # Hold the conversation open for the allotted time
        await asyncio.sleep(time_limit)
        logging.info(f"Time's up for question {i+1}, waiting for silence before moving on...")

        # Wait until both user and agent have been silent for 2 seconds before advancing
        silence_gap = 2.0
        while True:
            now = time.monotonic()
            last_speech = max(last_user_speech_at[0], last_agent_speech_at[0])
            silence_for = now - last_speech
            if silence_for >= silence_gap:
                break
            await asyncio.sleep(silence_gap - silence_for)

        logging.info(f"Silence detected, moving on to question {i+2}...")

        # Collect everything the user said during this question's window
        window = conversation_log[log_start:]
        user_turns = [text for role, text in window if role == "user"]

        dialogue.append({
            "interviewer_question": q_text,
            "user_answer": " ".join(user_turns),
        })

        
    # Wrap up
    await session.generate_reply(
        instructions="Let them know the interview is now complete."
    )

    # Persist the formatted dialogue
    output_path = "interview_output.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"dialogue": dialogue}, f, indent=2, ensure_ascii=False)

    print(f"Interview complete. Dialogue saved to {output_path}")


if __name__ == "__main__":
    agents.cli.run_app(server)
