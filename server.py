import os, json, uuid, logging, asyncio
from datetime import datetime
from enum import Enum
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse, Response
from dotenv import load_dotenv

import google.generativeai as genai

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s"
)

log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TWILIO_SID   = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_FROM  = os.getenv("TWILIO_FROM_NUMBER", "")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

BASE_URL   = os.getenv("BASE_URL", "http://localhost:8000")
LEADS_FILE = os.getenv("LEADS_FILE", "leads.json")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── Data helpers ──────────────────────────────────────────────────────────────

class Status(str, Enum):
    PENDING   = "pending"
    CALLING   = "calling"
    YES       = "yes"
    NO        = "no"
    MAYBE     = "maybe"
    VOICEMAIL = "voicemail"
    FAILED    = "failed"


def load_leads() -> dict:
    if Path(LEADS_FILE).exists():
        return json.loads(Path(LEADS_FILE).read_text())
    return {}


def save_leads(data: dict):
    Path(LEADS_FILE).write_text(
        json.dumps(data, indent=2, default=str)
    )


def update_lead(lead_id: str, **kwargs):
    data = load_leads()

    if lead_id in data:
        data[lead_id].update(kwargs)
        data[lead_id]["updated_at"] = datetime.utcnow().isoformat()
        save_leads(data)

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="AI Caller MVP")

# Serve single HTML file
@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(Path("index.html").read_text())

# ── Lead management API ───────────────────────────────────────────────────────

@app.get("/api/leads")
async def get_leads(status: Optional[str] = None):
    data = load_leads()

    leads = list(data.values())

    if status:
        leads = [l for l in leads if l["status"] == status]

    leads.sort(
        key=lambda x: x.get("created_at", ""),
        reverse=True
    )

    return leads


@app.post("/api/leads")
async def add_lead(request: Request):
    body = await request.json()

    leads = load_leads()

    lead_id = str(uuid.uuid4())[:8]

    lead = {
        "id": lead_id,
        "name": body.get("name", "").strip(),
        "phone": body.get("phone", "").strip(),
        "source": body.get("source", "Webinar"),

        "status": Status.PENDING,

        "call_sid": None,
        "transcript": None,
        "summary": None,

        "created_at": datetime.utcnow().isoformat(),
        "called_at": None,
        "updated_at": None,
    }

    leads[lead_id] = lead

    save_leads(leads)

    log.info(
        f"Lead added: {lead['name']} ({lead['phone']})"
    )

    return lead


@app.post("/api/leads/import")
async def import_leads(request: Request):
    """
    Bulk import:
    expects JSON array:
    [{name, phone, source}, ...]
    """

    body = await request.json()

    leads = load_leads()

    added = []

    for row in body:

        lead_id = str(uuid.uuid4())[:8]

        lead = {
            "id": lead_id,
            "name": row.get("name", "").strip(),
            "phone": row.get("phone", "").strip(),
            "source": row.get("source", "Webinar"),

            "status": Status.PENDING,

            "call_sid": None,
            "transcript": None,
            "summary": None,

            "created_at": datetime.utcnow().isoformat(),
            "called_at": None,
            "updated_at": None,
        }

        leads[lead_id] = lead
        added.append(lead)

    save_leads(leads)

    log.info(f"Bulk imported {len(added)} leads")

    return {
        "imported": len(added)
    }


@app.delete("/api/leads/{lead_id}")
async def delete_lead(lead_id: str):

    leads = load_leads()

    if lead_id in leads:
        del leads[lead_id]
        save_leads(leads)

    return {"ok": True}

# ── Calling API ───────────────────────────────────────────────────────────────

@app.post("/api/call/{lead_id}")
async def call_lead(
    lead_id: str,
    background_tasks: BackgroundTasks
):
    """
    Trigger a single outbound call.
    """

    leads = load_leads()

    lead = leads.get(lead_id)

    if not lead:
        return JSONResponse(
            {"error": "Lead not found"},
            status_code=404
        )

    background_tasks.add_task(_place_call, lead)

    return {
        "status": "calling",
        "lead": lead["name"]
    }


@app.post("/api/call-all")
async def call_all(background_tasks: BackgroundTasks):
    """
    Trigger calls for all pending leads.
    """

    leads = load_leads()

    pending = [
        l for l in leads.values()
        if l["status"] == Status.PENDING
    ]

    for lead in pending:
        background_tasks.add_task(_place_call, lead)

        # slight stagger to avoid Twilio rate limits
        await asyncio.sleep(0.5)

    return {
        "triggered": len(pending)
    }


async def _place_call(lead: dict):
    """
    Places the outbound Twilio call.
    """

    if not all([TWILIO_SID, TWILIO_TOKEN, TWILIO_FROM]):

        log.error(
            "Twilio credentials missing "
            "— set TWILIO_ACCOUNT_SID, "
            "TWILIO_AUTH_TOKEN, "
            "TWILIO_FROM_NUMBER in .env"
        )

        update_lead(
            lead["id"],
            status=Status.FAILED
        )

        return

    from twilio.rest import Client

    client = Client(
        TWILIO_SID,
        TWILIO_TOKEN
    )

    first_name = lead["name"].split()[0]

    webhook = (
        f"{BASE_URL}/voice/outbound"
        f"?lead_id={lead['id']}"
        f"&lead_name={first_name}"
    )

    try:

        call = client.calls.create(
            to=lead["phone"],
            from_=TWILIO_FROM,
            url=webhook,
            method="POST",
            timeout=30,
        )

        update_lead(
            lead["id"],
            status=Status.CALLING,
            call_sid=call.sid,
            called_at=datetime.utcnow().isoformat()
        )

        log.info(
            f"📞 Calling {lead['name']} "
            f"({lead['phone']}) "
            f"— SID: {call.sid}"
        )

    except Exception as e:

        log.error(
            f"Call failed for {lead['name']}: {e}"
        )

        update_lead(
            lead["id"],
            status=Status.FAILED
        )

# ── Twilio Voice Webhooks ─────────────────────────────────────────────────────

@app.post("/voice/outbound")
async def voice_outbound(
    request: Request,
    lead_id: str = "",
    lead_name: str = "",
):
    """
    Twilio hits this when lead picks up.
    """

    from twilio.twiml.voice_response import VoiceResponse, Gather

    r = VoiceResponse()

    script = (
        f"Hi {lead_name}, "
        f"this is an AI assistant calling on behalf of PropertyLab. "
        f"You registered for our webinar before. "
        f"Are you available for a quick call in the next 5 minutes? "
        f"Please say yes or no after the tone."
    )

    gather = Gather(
        input="speech",
        timeout=8,
        speech_timeout="auto",

        action=(
            f"{BASE_URL}/voice/response"
            f"?lead_id={lead_id}"
            f"&lead_name={lead_name}"
        ),

        method="POST",

        language="en-MY",
    )

    gather.say(
        script,
        voice="Polly.Joanna",
        language="en-US"
    )

    r.append(gather)

    # no speech detected
    r.redirect(
        f"{BASE_URL}/voice/no-answer"
        f"?lead_id={lead_id}"
        f"&lead_name={lead_name}",
        method="POST"
    )

    return Response(
        content=str(r),
        media_type="application/xml"
    )


@app.post("/voice/response")
async def voice_response(
    background_tasks: BackgroundTasks,

    lead_id: str = "",
    lead_name: str = "",

    SpeechResult: str = Form(default=""),
    Confidence: str = Form(default="0"),
    CallSid: str = Form(default=""),
):
    """
    Lead spoke.
    Respond immediately.
    Run AI classification in background.
    """

    from twilio.twiml.voice_response import VoiceResponse

    log.info(
        f"Response from {lead_name}: "
        f"'{SpeechResult}' "
        f"(conf={Confidence})"
    )

    r = VoiceResponse()

    r.say(
        "Thank you for your response. "
        "Our team will be in touch soon. Goodbye!",
        voice="Polly.Joanna"
    )

    r.hangup()

    background_tasks.add_task(
        _classify_and_save,

        lead_id=lead_id,
        speech=SpeechResult,
        call_sid=CallSid,
    )

    return Response(
        content=str(r),
        media_type="application/xml"
    )


@app.post("/voice/no-answer")
async def voice_no_answer(
    background_tasks: BackgroundTasks,

    lead_id: str = "",
    lead_name: str = "",

    CallSid: str = Form(default=""),
):
    """
    No speech detected.
    """

    from twilio.twiml.voice_response import VoiceResponse

    r = VoiceResponse()

    r.say(
        f"Hi, this is a message for {lead_name}. "
        "We are calling about our webinar. "
        "Please expect a callback from our team. "
        "Thank you!",
        voice="Polly.Joanna",
    )

    r.hangup()

    background_tasks.add_task(
        _classify_and_save,

        lead_id=lead_id,
        speech="",
        call_sid=CallSid,

        is_voicemail=True,
    )

    return Response(
        content=str(r),
        media_type="application/xml"
    )

# ── Gemini Intent Classification ──────────────────────────────────────────────

async def _classify_and_save(
    lead_id: str,
    speech: str,
    call_sid: str,
    is_voicemail: bool = False
):
    """
    Runs after call ends.
    Classifies lead intent with Gemini.
    """

    if is_voicemail or not speech.strip():

        update_lead(
            lead_id,

            status=Status.VOICEMAIL,
            transcript="(no response)",
            summary="Voicemail or no answer"
        )

        log.info(f"[{lead_id}] → VOICEMAIL")

        return

    if not GEMINI_API_KEY:

        log.warning(
            "No GEMINI_API_KEY "
            "— skipping intent classification"
        )

        update_lead(
            lead_id,

            status=Status.MAYBE,
            transcript=speech,
            summary=speech
        )

        return

    prompt = f"""
A lead was asked:

"Are you available for a quick call in the next 5 minutes?"

They responded:

"{speech}"

Classify as ONE of:
YES
NO
MAYBE
VOICEMAIL

Then write a short 1-sentence summary.

Return ONLY valid JSON.

Example:
{{
  "intent": "YES",
  "summary": "Lead is available now."
}}
"""

    try:

        model = genai.GenerativeModel(
            "gemini-1.5-flash"
        )

        response = model.generate_content(prompt)

        raw = response.text.strip()

        # Gemini sometimes wraps output in markdown
        if raw.startswith("```"):

            raw = (
                raw
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

        result = json.loads(raw)

        intent = (
            result
            .get("intent", "MAYBE")
            .upper()
        )

        summary = result.get("summary", speech)

    except Exception as e:

        log.error(
            f"Gemini classification failed: {e}"
        )

        intent  = "MAYBE"
        summary = speech

    status_map = {
        "YES": Status.YES,
        "NO": Status.NO,
        "MAYBE": Status.MAYBE,
        "VOICEMAIL": Status.VOICEMAIL,
    }

    final_status = status_map.get(
        intent,
        Status.MAYBE
    )

    update_lead(
        lead_id,

        status=final_status,
        transcript=speech,
        summary=summary
    )

    log.info(
        f"[{lead_id}] → {intent} | {summary}"
    )

# ── Stats endpoint ────────────────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():

    leads = load_leads()

    total = len(leads)

    by_status = {}

    for l in leads.values():

        s = l["status"]

        by_status[s] = (
            by_status.get(s, 0) + 1
        )

    return {
        "total": total,

        "pending":   by_status.get("pending", 0),
        "calling":   by_status.get("calling", 0),

        "yes":       by_status.get("yes", 0),
        "no":        by_status.get("no", 0),
        "maybe":     by_status.get("maybe", 0),

        "voicemail": by_status.get("voicemail", 0),
        "failed":    by_status.get("failed", 0),
    }

# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "server:app",

        host="0.0.0.0",
        port=8000,

        reload=True
    )