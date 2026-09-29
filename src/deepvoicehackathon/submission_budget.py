"""Pure submission-quota arithmetic with explicit counts and KST calendar days.

The deadline is exclusive. Every positive-duration future calendar window has
the full daily quota as an upper bound, even when the final day is partial.
These are quota ceilings, not promises that upload/evaluation time will fit.
No usage is inferred from submission IDs, timestamps, or the wall clock.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta, timezone

KST = timezone(timedelta(hours=9), name="KST")


def _kst(value: datetime, name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{name} must be a timezone-aware datetime")
    return value.astimezone(KST)


def _count(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def calculate_submission_budget(
    *, now: datetime, deadline: datetime, daily_limit: int, used_today: int,
    campaign_limit: int | None = None, campaign_used: int = 0,
) -> dict:
    """Return a JSON-ready snapshot without reading or changing external state.

    ``used_today`` belongs to the KST calendar date containing ``now``.
    Platform availability and campaign allowance are reported separately;
    With an explicit campaign limit, ``effective_remaining`` is their minimum.
    Without one, it equals the platform ceiling and campaign remaining is null.
    ``next_reset_kst`` is null if no reset occurs strictly before the deadline.
    Campaign allowance is retained after closure, but effective availability is
    zero because the platform window has ended.
    """
    now_kst, deadline_kst = _kst(now, "now"), _kst(deadline, "deadline")
    daily_limit = _count(daily_limit, "daily_limit")
    used_today = _count(used_today, "used_today")
    if campaign_limit is not None:
        campaign_limit = _count(campaign_limit, "campaign_limit")
    campaign_used = _count(campaign_used, "campaign_used")
    if used_today > daily_limit:
        raise ValueError("used_today must not exceed daily_limit")
    if campaign_limit is not None and campaign_used > campaign_limit:
        raise ValueError("campaign_used must not exceed campaign_limit")

    closed = now_kst >= deadline_kst
    today_available = 0 if closed else daily_limit - used_today
    full_dates, partial_windows = [], []
    next_reset = None
    if not closed:
        cursor = datetime.combine(now_kst.date() + timedelta(days=1), time.min, KST)
        if cursor < deadline_kst:
            next_reset = cursor
        while cursor < deadline_kst:
            # If the deadline's local date is later, this entire day is open.
            if cursor.date() < deadline_kst.date():
                full_dates.append(cursor.date().isoformat())
                cursor += timedelta(days=1)
            else:
                partial_windows.append({
                    "date": cursor.date().isoformat(),
                    "start_kst": cursor.isoformat(),
                    "end_kst": deadline_kst.isoformat(),
                    "quota_ceiling": daily_limit,
                })
                break

    future_ceiling = daily_limit * (len(full_dates) + len(partial_windows))
    platform_ceiling = today_available + future_ceiling
    campaign_remaining = None if campaign_limit is None else campaign_limit - campaign_used
    return {
        "now_kst": now_kst.isoformat(),
        "deadline_kst": deadline_kst.isoformat(),
        "deadline_exclusive": True,
        "closed": closed,
        "daily_limit": daily_limit,
        "used_today": used_today,
        "current_day_kst": now_kst.date().isoformat(),
        "current_day_available": today_available,
        "next_reset_kst": next_reset.isoformat() if next_reset is not None else None,
        "future_full_window_dates": full_dates,
        "future_full_window_count": len(full_dates),
        "future_full_window_ceiling": daily_limit * len(full_dates),
        "future_partial_windows": partial_windows,
        "future_partial_window_count": len(partial_windows),
        "future_platform_ceiling": future_ceiling,
        "remaining_platform_ceiling": platform_ceiling,
        "campaign_limit": campaign_limit,
        "internal_limit_active": campaign_limit is not None,
        "campaign_used": campaign_used,
        "campaign_remaining": campaign_remaining,
        "effective_remaining": (platform_ceiling if campaign_remaining is None
                                else min(platform_ceiling, campaign_remaining)),
        "notes": [
            "Usage counts are explicit inputs; no submission timestamps or IDs are interpreted.",
            "Current-day availability already subtracts used_today; future windows exclude today.",
            "Every positive-duration future day receives at most daily_limit, including a partial final day.",
            ("No internal campaign limit is active; effective_remaining equals the platform ceiling."
             if campaign_limit is None else
             "Platform ceilings and campaign allowance are distinct; effective_remaining is their minimum."),
            "Quota arithmetic does not model upload duration, evaluation latency, or service availability.",
        ],
    }
