from __future__ import annotations

import itertools
import random
from datetime import datetime, timedelta
from typing import Any


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

PROVIDERS = [
    {"id": "dr-patel", "name": "Dr. Anita Patel", "specialty": "General Practice"},
    {"id": "dr-nguyen", "name": "Dr. Kevin Nguyen", "specialty": "Dermatology"},
    {"id": "dr-osei", "name": "Dr. Femi Osei", "specialty": "Cardiology"},
]

_booking_id_counter = itertools.count(1000)

# Populated by reset_state()
_slots: dict[str, dict[str, Any]] = {}
_bookings: dict[str, dict[str, Any]] = {}


def _gen_slots_for_provider(provider_id: str, start_date: datetime, days: int = 7) -> None:
    """Generate a handful of open slots per weekday for a provider."""
    for day_offset in range(days):
        day = start_date + timedelta(days=day_offset)
        if day.weekday() >= 5:  # skip weekends
            continue
        for hour in (9, 11, 14, 16):
            slot_id = f"{provider_id}-{day.strftime('%Y%m%d')}-{hour:02d}00"
            _slots[slot_id] = {
                "slot_id": slot_id,
                "provider_id": provider_id,
                "date": day.strftime("%Y-%m-%d"),
                "time": f"{hour:02d}:00",
                "status": "open",  # open | booked
            }


def reset_state(seed: int = 42) -> None:
    """Reset all mock data to a known baseline. Call before each scenario/eval run."""
    global _slots, _bookings, _booking_id_counter
    random.seed(seed)
    _slots = {}
    _bookings = {}
    _booking_id_counter = itertools.count(1000)

    anchor = datetime(2026, 9, 28)  # a Monday, fixed so eval runs are deterministic
    for provider in PROVIDERS:
        _gen_slots_for_provider(provider["id"], anchor)

    # Pre-book a couple of slots so "fully booked" / conflict scenarios are possible.
    preseeded = [s for s in _slots.values() if s["time"] == "09:00"][:2]
    for slot in preseeded:
        slot["status"] = "booked"
        bid = f"appt-{next(_booking_id_counter)}"
        _bookings[bid] = {
            "booking_id": bid,
            "slot_id": slot["slot_id"],
            "provider_id": slot["provider_id"],
            "patient_name": "Existing Patient",
            "date": slot["date"],
            "time": slot["time"],
            "status": "confirmed",
        }


# Initialize on import so ad-hoc scripts work without an explicit reset.
reset_state()


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def check_slots(
    provider_name: str | None = None,
    specialty: str | None = None,
    date: str | None = None,
) -> dict[str, Any]:
    """
    List open appointment slots, optionally filtered by provider name,
    specialty, and/or date (YYYY-MM-DD).

    Returns:
        {"slots": [ {slot_id, provider_id, provider_name, specialty, date, time}, ... ]}
        or {"error": "..."} if a filter matches nothing meaningful (e.g. unknown provider).
    """
    matched_providers = PROVIDERS
    if provider_name:
        needle = provider_name.strip().lower()
        matched_providers = [
            p for p in PROVIDERS
            if needle in p["name"].lower() or needle in p["id"].lower()
        ]
        if not matched_providers:
            return {"error": f"No provider found matching '{provider_name}'."}

    if specialty:
        needle = specialty.strip().lower()
        matched_providers = [
            p for p in matched_providers if needle in p["specialty"].lower()
        ]
        if not matched_providers:
            return {"error": f"No provider found with specialty '{specialty}'."}

    provider_ids = {p["id"] for p in matched_providers}
    provider_lookup = {p["id"]: p for p in PROVIDERS}

    results = []
    for slot in _slots.values():
        if slot["status"] != "open":
            continue
        if slot["provider_id"] not in provider_ids:
            continue
        if date and slot["date"] != date:
            continue
        provider = provider_lookup[slot["provider_id"]]
        results.append(
            {
                "slot_id": slot["slot_id"],
                "provider_id": provider["id"],
                "provider_name": provider["name"],
                "specialty": provider["specialty"],
                "date": slot["date"],
                "time": slot["time"],
            }
        )

    results.sort(key=lambda s: (s["date"], s["time"], s["provider_id"]))

    if not results:
        return {"slots": [], "message": "No open slots match those criteria."}
    return {"slots": results}


def book_appointment(
    slot_id: str,
    patient_name: str,
    reason: str | None = None,
) -> dict[str, Any]:
    """
    Book an open slot for a patient. Fails if the slot doesn't exist or is
    already booked (mirrors a real scheduling system's race/conflict case).

    Returns:
        {"booking_id", "slot_id", "provider_id", "date", "time", "patient_name", "status"}
        or {"error": "..."}
    """
    if not patient_name or not patient_name.strip():
        return {"error": "patient_name is required to book an appointment."}

    slot = _slots.get(slot_id)
    if slot is None:
        return {"error": f"Slot '{slot_id}' does not exist. Use check_slots to find a valid slot_id."}
    if slot["status"] != "open":
        return {"error": f"Slot '{slot_id}' is no longer available. Please choose another slot."}

    slot["status"] = "booked"
    booking_id = f"appt-{next(_booking_id_counter)}"
    booking = {
        "booking_id": booking_id,
        "slot_id": slot_id,
        "provider_id": slot["provider_id"],
        "date": slot["date"],
        "time": slot["time"],
        "patient_name": patient_name.strip(),
        "reason": reason or "",
        "status": "confirmed",
    }
    _bookings[booking_id] = booking
    return booking


def cancel_appointment(booking_id: str) -> dict[str, Any]:
    """
    Cancel an existing booking and free up its slot.

    Returns:
        {"booking_id", "status": "cancelled"} or {"error": "..."}
    """
    booking = _bookings.get(booking_id)
    if booking is None:
        return {"error": f"Booking '{booking_id}' does not exist."}
    if booking["status"] == "cancelled":
        return {"error": f"Booking '{booking_id}' is already cancelled."}

    booking["status"] = "cancelled"
    slot = _slots.get(booking["slot_id"])
    if slot is not None:
        slot["status"] = "open"
    return {"booking_id": booking_id, "status": "cancelled"}


# ---------------------------------------------------------------------------
# Tool schemas (Anthropic Messages API `tools` format)
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "check_slots",
        "description": (
            "Look up open appointment slots. Filter by provider name, medical "
            "specialty, and/or a specific date (YYYY-MM-DD). Call this before "
            "booking to find a valid slot_id, and whenever the patient asks "
            "what's available."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "provider_name": {
                    "type": "string",
                    "description": "Full or partial provider name, e.g. 'Dr. Patel'.",
                },
                "specialty": {
                    "type": "string",
                    "description": "Medical specialty, e.g. 'Cardiology'.",
                },
                "date": {
                    "type": "string",
                    "description": "Specific date to check, in YYYY-MM-DD format.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "book_appointment",
        "description": (
            "Book a specific open slot for a named patient. Requires a slot_id "
            "obtained from check_slots — never guess a slot_id. Fails if the "
            "slot is already booked."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "slot_id": {
                    "type": "string",
                    "description": "The exact slot_id returned by check_slots.",
                },
                "patient_name": {
                    "type": "string",
                    "description": "Full name of the patient the appointment is for.",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief reason for the visit, if the patient provided one.",
                },
            },
            "required": ["slot_id", "patient_name"],
        },
    },
    {
        "name": "cancel_appointment",
        "description": (
            "Cancel an existing confirmed appointment by its booking_id. "
            "Use this when a patient wants to cancel or reschedule (cancel "
            "then book a new slot)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "booking_id": {
                    "type": "string",
                    "description": "The booking_id to cancel, e.g. 'appt-1001'.",
                },
            },
            "required": ["booking_id"],
        },
    },
]

TOOL_IMPLEMENTATIONS = {
    "check_slots": check_slots,
    "book_appointment": book_appointment,
    "cancel_appointment": cancel_appointment,
}
