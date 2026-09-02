"""
Stage 5 - Messaging layer.

IMPORTANT BOUNDARY: this module never decides WHETHER or HOW OFTEN to contact
a customer - policy.py has already made and justified that decision. This
module only phrases an already-sanctioned action into customer-facing text.
That boundary is enforced in code, first thing, before even checking for an
API key: 'compliance_stop' and 'give_up_unlikely' always return None.
"""
import os

NO_CONTACT_ACTIONS = {"compliance_stop", "give_up_unlikely"}

SYSTEM_PROMPT = (
    "You write short customer-facing payment-recovery messages for a fintech app. "
    "Rules you must always follow: "
    "1) At most 2 sentences. "
    "2) Non-threatening tone - no urgency or pressure language (never say things like "
    "'immediately', 'act now', 'your account will be suspended'). "
    "3) Write in the requested language style exactly: 'en' means plain English; "
    "'hi' means Hinglish - a natural Hindi/English mix written in Roman script, the way "
    "Indian fintech apps actually message users. "
    "4) Never invent extra fees, deadlines, or threats not given to you. "
    "Return only the message text, nothing else."
)

# Offline fallback templates, keyed by (action, lang). Used whenever
# ANTHROPIC_API_KEY is not set, so the pipeline always runs end-to-end.
TEMPLATES = {
    ("retry_0h", "en"): "We're retrying your recent payment of Rs.{amount:.0f} right now. No action is needed from you.",
    ("retry_0h", "hi"): "Aapka Rs.{amount:.0f} ka payment hum abhi retry kar rahe hain. Aapko kuch karne ki zaroorat nahi hai.",
    ("retry_4h", "en"): "We'll automatically retry your payment of Rs.{amount:.0f} in a few hours. No action is needed from you.",
    ("retry_4h", "hi"): "Aapka Rs.{amount:.0f} ka payment hum kuch ghanton mein dobara try karenge. Kuch karne ki zaroorat nahi.",
    ("retry_24h", "en"): "We'll automatically retry your payment of Rs.{amount:.0f} tomorrow. No action is needed from you.",
    ("retry_24h", "hi"): "Aapka Rs.{amount:.0f} ka payment hum kal dobara try karenge. Aapko kuch karne ki zaroorat nahi hai.",
    ("send_reminder_alt_method", "en"): "Your payment of Rs.{amount:.0f} didn't go through. Whenever convenient, you could try a different payment method.",
    ("send_reminder_alt_method", "hi"): "Aapka Rs.{amount:.0f} ka payment complete nahi ho paya. Jab convenient ho, ek alag payment method try kar sakte hain.",
    ("escalate_human", "en"): "We noticed an issue with your recent payment of Rs.{amount:.0f}. A member of our team will reach out to help whenever suits you.",
    ("escalate_human", "hi"): "Aapke Rs.{amount:.0f} ke payment mein kuch issue aaya hai. Hamari team aapse jald hi contact karegi, jab bhi aapko suitable ho.",
}


def _fallback_message(action: str, lang: str, amount: float) -> str:
    template = TEMPLATES.get((action, lang)) or TEMPLATES.get((action, "en"))
    return template.format(amount=amount)


def _anthropic_message(action: str, lang: str, amount: float, failure_code: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    lang_label = "English" if lang == "en" else "Hinglish (Hindi/English mix, Roman script)"
    user_prompt = (
        f"Action already decided: {action}. Failure reason (internal, do not mention codes "
        f"verbatim to the customer): {failure_code}. Payment amount: Rs.{amount:.0f}. "
        f"Write the message in {lang_label}."
    )
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=150,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text.strip()


def generate_message(action: str, lang: str, amount: float, failure_code: str = "") -> str:
    """Return customer-facing text for an already-decided action, or None if
    the action is a stop/give-up (no contact should ever be made)."""
    if action in NO_CONTACT_ACTIONS:
        return None

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _anthropic_message(action, lang, amount, failure_code)
        except Exception as exc:  # network/API issues shouldn't kill the pipeline
            print(f"[messenger] Anthropic call failed ({exc}), falling back to template.")
            return _fallback_message(action, lang, amount)

    return _fallback_message(action, lang, amount)
