"""Checkpoint management for resumable workflows."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.state import GraphState

logger = get_logger("core.checkpoints")


class CheckpointManager:
    """Save/load workflow snapshots to `.daedalus/checkpoints`."""

    def _checkpoint_dir(self, repo_root: str = "") -> Path:
        settings = get_settings()
        base = Path(repo_root or settings.target_repo_path or ".").resolve()
        path = base / ".daedalus" / "checkpoints"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def save_checkpoint(self, state: GraphState, checkpoint_type: str = "auto", repo_root: str = "") -> str:
        checkpoint_id = f"{checkpoint_type}_{uuid.uuid4().hex[:8]}"
        checkpoint_dir = self._checkpoint_dir(repo_root or state.repo_root)
        checkpoint_file = checkpoint_dir / f"{checkpoint_id}.json"

        state_payload = state.model_dump()
        state_payload["checkpoint_id"] = state.checkpoint_id
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "checkpoint_type": checkpoint_type,
            "timestamp": datetime.now(UTC).isoformat(),
            "state": state_payload,
        }

        checkpoint_file.write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")
        (checkpoint_dir / "latest.json").write_text(json.dumps(checkpoint, indent=2, default=str), encoding="utf-8")

        state.checkpoint_id = checkpoint_id
        state.last_checkpoint_path = str(checkpoint_file)
        logger.info("Checkpoint saved: %s", checkpoint_id)
        return checkpoint_id

    def load_checkpoint(self, checkpoint_id: str | None = None, repo_root: str = "") -> GraphState | None:
        checkpoint_dir = self._checkpoint_dir(repo_root)
        checkpoint_file = checkpoint_dir / f"{checkpoint_id}.json" if checkpoint_id else checkpoint_dir / "latest.json"

        if not checkpoint_file.exists():
            return None

        try:
            payload = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            state_payload = payload.get("state", {})
            state = GraphState.from_dict(state_payload)
            state.resumed_from_checkpoint = True
            state.checkpoint_id = payload.get("checkpoint_id")
            state.last_checkpoint_path = str(checkpoint_file)
            logger.info("Checkpoint loaded: %s", state.checkpoint_id)
            return state
        except Exception as exc:
            logger.error("Failed to load checkpoint %s: %s", checkpoint_file, exc)
            return None

    def list_checkpoints(self, repo_root: str = "") -> list[dict]:
        checkpoint_dir = self._checkpoint_dir(repo_root)
        items: list[dict] = []
        for path in checkpoint_dir.glob("*.json"):
            if path.name == "latest.json":
                continue
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                items.append(
                    {
                        "checkpoint_id": payload.get("checkpoint_id"),
                        "type": payload.get("checkpoint_type"),
                        "timestamp": payload.get("timestamp"),
                    }
                )
            except Exception:
                continue
        items.sort(key=lambda row: str(row.get("timestamp", "")), reverse=True)
        return items

    def mark_latest_approval(self, approved: bool, repo_root: str = "") -> bool:
        checkpoint_dir = self._checkpoint_dir(repo_root)
        latest_file = checkpoint_dir / "latest.json"
        if not latest_file.exists():
            return False
        try:
            payload = json.loads(latest_file.read_text(encoding="utf-8"))
            state_payload = payload.get("state", {})
            pending = state_payload.setdefault("pending_approval", {})
            pending["approved"] = approved
            state_payload["needs_human_approval"] = not approved
            payload["state"] = state_payload
            latest_file.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            checkpoint_id = payload.get("checkpoint_id")
            if checkpoint_id:
                specific = checkpoint_dir / f"{checkpoint_id}.json"
                if specific.exists():
                    specific.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
            return True
        except Exception as exc:
            logger.error("Failed to mark latest checkpoint approval: %s", exc)
            return False


checkpoint_manager = CheckpointManager()
