#TODO: FIND A WAY TO INTEGRATE TIME LIMIT PER QUESTION / FOLLOW UP QUESTIONS WHILE KEEPING THE INTERVIEW EXPERIENCE CLEAN / NOT BUGGY

"""
interview.py
------------
A LiveKit voice agent that conducts a structured interview with a participant.

Questions are loaded from interview_test.json. For each question the agent:
  1. Speaks the question aloud via TTS.
  2. Listens for the participant's response.
  3. Advances to the next question once the participant has been silent for
     5 seconds (controlled by min/max_endpointing_delay).

After all questions are answered the dialogue is saved to interview_output.json
and the LiveKit room connection is closed.
"""

import asyncio
import json
import logging
from dotenv import load_dotenv

from livekit import agents, rtc
from livekit.agents import AgentServer, AgentSession, Agent, room_io, TurnHandlingOptions
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

load_dotenv(".env")

# Load interview questions from disk at startup.
# Each entry is expected to have at least a "question" field.
with open("interview_test.json") as f:
    QUESTIONS = json.load(f)["questions"]


class InterviewAgent(Agent):
    """
    A passive interviewer agent.

    Its sole responsibility is to ask the question it is given and listen.
    It does not generate follow-up questions or deviate from the script —
    question sequencing is handled by the session loop in interview_agent().
    """

    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a friendly interviewer. Ask the question given to you and listen to the participant's answer.
            Do not ask follow-up questions or deviate from the current question."""
        )


server = AgentServer()


@server.rtc_session(agent_name="interview-agent")
async def interview_agent(ctx: agents.JobContext):
    """
    Main session handler. Runs once per connected participant.

    Orchestrates the full interview flow:
      - Starts the LiveKit agent session with STT, LLM, and TTS.
      - Iterates through each question, blocking until the participant finishes
        their answer (detected via a 5-second silence window).
      - Collects all spoken turns into a structured dialogue.
      - Persists the dialogue to interview_output.json.
      - Disconnects from the room when complete.
    """

    # Full chronological log of all spoken turns as (role, text) tuples.
    conversation_log: list[tuple[str, str]] = []

    # Signals that the participant has committed at least one speech turn,
    # allowing the loop to advance to the next question.
    user_answered = asyncio.Event()

    session = AgentSession(
        stt="deepgram/nova-3:multi",        # Speech-to-text
        llm="openai/gpt-4.1-mini",          # Language model for agent responses
        tts="deepgram/aura-2-athena-en",    # Text-to-speech voice
        vad=silero.VAD.load(),              # Voice activity detection
        turn_handling=TurnHandlingOptions(
            turn_detection=MultilingualModel(),
            # Require 5 seconds of continuous silence before committing a turn.
            # This gives participants time to pause mid-thought without the
            # agent cutting them off prematurely.
            min_endpointing_delay=5.0,
            max_endpointing_delay=5.0,
            interruption={"enabled": False},
        ),
    )

    @session.on("agent_speech_committed")
    def _on_agent_speech(ev):
        """Logs and records every finalized agent utterance."""
        logging.info(f"Agent speech committed: {ev}")
        text = getattr(ev, "agent_transcript", None) or getattr(ev, "transcript", None) or str(ev)
        conversation_log.append(("agent", text))

    @session.on("user_input_transcribed")
    def _on_user_speech(ev):
        """
        Fires whenever the STT pipeline produces a transcript.
        Intermediate (non-final) results are ignored — only the finalized
        transcript is recorded and used to signal that the participant answered.
        """
        if not ev.is_final:
            return
        logging.info(f"User input transcribed: {ev}")
        conversation_log.append(("user", ev.transcript))
        user_answered.set()

    await session.start(
        room=ctx.room,
        agent=InterviewAgent(),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                # Use telephony-grade noise cancellation for SIP participants,
                # standard BVC for all others.
                noise_cancellation=lambda params: (
                    noise_cancellation.BVCTelephony()
                    if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                    else noise_cancellation.BVC()
                ),
            ),
        ),
    )

    await session.say("Hey, thanks for coming. Let's get started.")

    dialogue = []

    for question in QUESTIONS:
        q_text = question["question"]

        # Mark where in the log this question's window starts so we can
        # slice out only the turns that belong to this question later.
        log_start = len(conversation_log)

        # Reset the event before asking so a leftover signal from the
        # previous question doesn't immediately advance the loop.
        user_answered.clear()
        await session.say(q_text)

        # Block until the participant has finished their answer.
        # The 5-second endpointing delay means this resolves only after
        # sustained silence, not on the first word.
        await user_answered.wait()

        # Collect all user turns that occurred during this question's window.
        window = conversation_log[log_start:]
        user_turns = [text for role, text in window if role == "user"]

        dialogue.append({
            "interviewer_question": q_text,
            "user_answer": " ".join(user_turns),
        })

    await session.say("That's all the questions. Thanks for your time.")

    # Persist the structured dialogue for downstream processing.
    with open("interview_output.json", "w", encoding="utf-8") as f:
        json.dump({"dialogue": dialogue}, f, indent=2, ensure_ascii=False)

    # Cleanly close the LiveKit room connection.
    await ctx.room.disconnect()


if __name__ == "__main__":
    agents.cli.run_app(server)
