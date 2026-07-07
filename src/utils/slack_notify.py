import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


DEFAULT_THRESHOLD_SECONDS = 2
SUMMARY_KEYS = ("AC@1", "Avg@1", "AC@3", "Avg@3", "AC@5", "Avg@5")


def _load_dotenv() -> None:
    dotenv_path = Path(__file__).resolve().parents[2] / ".env"
    if not dotenv_path.is_file():
        return

    with dotenv_path.open("r") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export ") :].strip()
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


_load_dotenv()


def _format_timestamp(epoch_seconds: float) -> str:
    return datetime.fromtimestamp(epoch_seconds).astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _format_duration(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, remaining_seconds = divmod(total_seconds, 60)
    hours, remaining_minutes = divmod(minutes, 60)

    if hours:
        return f"{hours}h {remaining_minutes}m {remaining_seconds}s"
    if minutes:
        return f"{minutes}m {remaining_seconds}s"
    return f"{remaining_seconds}s"


def _load_json(path: str) -> Optional[Dict]:
    if not path or not os.path.isfile(path):
        return None
    with open(path, "r") as handle:
        return json.load(handle)


def _load_result_files(result_files: Sequence[str]) -> List[Dict]:
    records: List[Dict] = []
    for file_path in result_files:
        data = _load_json(file_path)
        if data:
            records.append(data)
    return records


def _aggregate_summary(records: Sequence[Dict]) -> Dict[str, Dict[str, float]]:
    grouped_metrics: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        dataset = record.get("dataset")
        metrics = record.get("metrics", {})
        if not dataset or not isinstance(metrics, dict):
            continue
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                grouped_metrics[dataset][metric_name].append(float(metric_value))

    summary: Dict[str, Dict[str, float]] = {}
    for dataset, metrics in grouped_metrics.items():
        summary[dataset] = {
            metric_name: round(sum(values) / len(values), 4)
            for metric_name, values in metrics.items()
            if values
        }
    return summary


def _detect_model_used(records: Sequence[Dict], fallback: str = "unknown") -> str:
    for record in records:
        model_used = record.get("model_used")
        if model_used:
            return str(model_used)
    return fallback


def _build_summary_lines(summary: Dict[str, Dict[str, float]]) -> List[str]:
    if not summary:
        return ["- No result summary available yet."]

    lines: List[str] = []
    for dataset in sorted(summary.keys()):
        metrics = summary[dataset]
        metric_text = []
        for key in SUMMARY_KEYS:
            if key in metrics:
                metric_text.append(f"{key} {metrics[key]:.4f}".rstrip("0").rstrip("."))
        if not metric_text:
            metric_text = [f"{key} {value:.4f}".rstrip("0").rstrip(".") for key, value in sorted(metrics.items())]
        lines.append(f"- {dataset}: {', '.join(metric_text)}")
    return lines


def build_message(
    *,
    command: str,
    start_epoch: float,
    end_epoch: float,
    exit_code: int,
    status: str,
    reason: str,
    result_files: Sequence[str],
    mention_user_id: str = "",
    total_time_override: Optional[float] = None,
) -> Tuple[str, float, Dict[str, Dict[str, float]], str]:
    records = _load_result_files(result_files)
    summary = _aggregate_summary(records)
    model_used = _detect_model_used(records)

    elapsed_seconds = total_time_override if total_time_override is not None else max(end_epoch - start_epoch, 0.0)
    start_text = _format_timestamp(start_epoch)
    end_text = _format_timestamp(end_epoch)
    duration_text = _format_duration(elapsed_seconds)

    status_text = status
    if exit_code == 0 and status == "completed":
        status_text = "completed"
    elif exit_code != 0 and status == "completed":
        status_text = "failed"

    lines = ["*Simulation Report*", 
                f"- Duration: {duration_text}", 
                f"- Start: {start_text}", 
                f"- End: {end_text}", 
                f"- Model: `{model_used}`",
                f"- Status: {status_text}", 
                f"- Exit code: {exit_code}", 
                f"- Command: `{command}`", 
            ]
    
    if reason:
        lines.append(f"- Note: {reason}")

    if result_files:
        lines.append(f"- Result files: {len(result_files)}")

    lines.append("")
    lines.append("*Result Summary*")
    lines.extend(_build_summary_lines(summary))

    message_body = "\n".join(lines)
    if mention_user_id:
        message_body = f"<@{mention_user_id}>\n{message_body}"

    return message_body, elapsed_seconds, summary, model_used


def send_slack_notification(webhook_url: str, message: str, timeout_seconds: int = 10) -> None:
    if not webhook_url:
        return

    payload = json.dumps({"text": message}, ensure_ascii=False).encode("utf-8")
    request = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={"Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )

    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        response.read()


def maybe_notify_slack(
    *,
    webhook_url: str,
    command: str,
    start_epoch: float,
    end_epoch: float,
    exit_code: int,
    status: str,
    reason: str,
    result_files: Sequence[str],
    mention_user_id: str = "",
    threshold_seconds: int = DEFAULT_THRESHOLD_SECONDS,
) -> bool:
    resolved_webhook = webhook_url or os.environ.get("SLACK_WEBHOOK_URL", "")
    elapsed_seconds = max(end_epoch - start_epoch, 0.0)
    if elapsed_seconds < threshold_seconds:
        # print(
        #     f"Slack notification skipped: elapsed {elapsed_seconds:.1f}s < threshold {threshold_seconds}s",
        #     file=sys.stderr,
        # )
        return False

    if not resolved_webhook:
        print("Slack notification skipped: webhook URL is not configured", file=sys.stderr)
        return False

    message, _, _, _ = build_message(
        command=command,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        exit_code=exit_code,
        status=status,
        reason=reason,
        result_files=result_files,
        mention_user_id=mention_user_id,
    )

    try:
        send_slack_notification(resolved_webhook, message)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as exc:
        print(f"Slack notification failed: {exc}", file=sys.stderr)
        return False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send a Slack notification for long-running simulations")
    parser.add_argument("--webhook-url", default=os.environ.get("SLACK_WEBHOOK_URL", ""))
    parser.add_argument("--mention-user-id", default=os.environ.get("SLACK_MENTION_USER_ID", ""))
    parser.add_argument("--command", required=True)
    parser.add_argument("--start-epoch", type=float, required=True)
    parser.add_argument("--end-epoch", type=float, required=True)
    parser.add_argument("--exit-code", type=int, required=True)
    parser.add_argument("--status", default="completed")
    parser.add_argument("--reason", default="")
    parser.add_argument("--threshold-seconds", type=int, default=DEFAULT_THRESHOLD_SECONDS)
    parser.add_argument("--result-file", action="append", default=[])
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    notified = maybe_notify_slack(
        webhook_url=args.webhook_url,
        command=args.command,
        start_epoch=args.start_epoch,
        end_epoch=args.end_epoch,
        exit_code=args.exit_code,
        status=args.status,
        reason=args.reason,
        result_files=args.result_file,
        mention_user_id=args.mention_user_id,
        threshold_seconds=args.threshold_seconds,
    )
    return 0 if notified else 0


if __name__ == "__main__":
    raise SystemExit(main())