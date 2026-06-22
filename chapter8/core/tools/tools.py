from datetime import datetime, timezone

def _now(args: dict) -> str:
    return datetime.now(timezone.utc).isoformat()


def _wordcount(args: dict) -> str:
    return str(len(args["text"].split()))
