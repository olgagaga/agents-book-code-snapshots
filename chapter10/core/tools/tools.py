from datetime import datetime, timezone
import time

def _now(args: dict) -> str:
    return datetime.now(timezone.utc).isoformat()


def _wordcount(args: dict) -> str:
    return str(len(args["text"].split()))

def _sleep_for(args: dict) -> str:
    seconds = float(args["seconds"])
    time.sleep(seconds)
    return f"slept for {seconds:.1f}s"