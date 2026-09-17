"""Write actions -- the orange box, and the reason it is drawn in orange.

A write action is the only part of this platform that can do something the user
cannot undo by closing the tab. The architecture diagram gives it one rectangle;
in practice it needs four properties, all enforced here:

1. **Approval.** By default a write is *proposed*, not executed. Something with
   authority -- a human, or a policy that has been reasoned about -- approves it.
2. **Idempotency.** Every execution carries a key. Replaying an approved write
   returns the first result instead of sending the email twice.
3. **Dry run.** Every action can be rehearsed and its effect described.
4. **Audit.** Proposed, approved, rejected and executed all get a record.

These are not wired into the request path. Nothing in the pipeline below calls
into this module -- an agent loop that can write has to be built deliberately.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from aie.types import new_id

logger = logging.getLogger("aie.actions.write")


class WriteStatus(str, Enum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"


@dataclass
class WriteProposal:
    id: str
    action: str
    arguments: dict[str, Any]
    tenant_id: str
    user_id: str
    idempotency_key: str
    status: WriteStatus = WriteStatus.PROPOSED
    preview: str = ""
    result: Any = None
    error: str | None = None
    created_at: float = field(default_factory=time.time)
    decided_by: str | None = None


@dataclass
class WriteAction:
    name: str
    description: str
    handler: Callable[..., Any]
    # describe() renders what *would* happen, for the approver to read.
    describe: Callable[..., str]
    requires_approval: bool = True


@dataclass
class AuditRecord:
    at: float
    proposal_id: str
    action: str
    status: WriteStatus
    actor: str
    detail: str = ""


class WriteActionRegistry:
    def __init__(self) -> None:
        self._actions: dict[str, WriteAction] = {}
        self._proposals: dict[str, WriteProposal] = {}
        self._by_idempotency: dict[str, str] = {}
        self.audit_log: list[AuditRecord] = []
        self._lock = threading.Lock()

    def register(self, action: WriteAction) -> None:
        self._actions[action.name] = action

    def names(self) -> list[str]:
        return sorted(self._actions)

    def _record(self, proposal: WriteProposal, actor: str, detail: str = "") -> None:
        record = AuditRecord(
            at=time.time(),
            proposal_id=proposal.id,
            action=proposal.action,
            status=proposal.status,
            actor=actor,
            detail=detail,
        )
        self.audit_log.append(record)
        logger.info(
            "write %s %s by %s %s", proposal.action, proposal.status.value, actor, detail
        )

    def propose(
        self,
        name: str,
        *,
        tenant_id: str,
        user_id: str,
        idempotency_key: str | None = None,
        **arguments: Any,
    ) -> WriteProposal:
        action = self._actions.get(name)
        if action is None:
            raise KeyError(f"no write action named {name!r}; have {self.names()}")

        key = idempotency_key or f"{name}:{tenant_id}:{sorted(arguments.items())}"
        with self._lock:
            existing_id = self._by_idempotency.get(key)
            if existing_id:
                return self._proposals[existing_id]

            proposal = WriteProposal(
                id=new_id("write"),
                action=name,
                arguments=arguments,
                tenant_id=tenant_id,
                user_id=user_id,
                idempotency_key=key,
                preview=action.describe(**arguments),
            )
            if not action.requires_approval:
                proposal.status = WriteStatus.APPROVED
                proposal.decided_by = "policy:no_approval_required"

            self._proposals[proposal.id] = proposal
            self._by_idempotency[key] = proposal.id

        self._record(proposal, actor=user_id, detail=proposal.preview)
        return proposal

    def approve(self, proposal_id: str, *, approver: str) -> WriteProposal:
        proposal = self._require(proposal_id)
        if proposal.status is not WriteStatus.PROPOSED:
            raise ValueError(f"proposal {proposal_id} is {proposal.status.value}, not proposed")
        proposal.status = WriteStatus.APPROVED
        proposal.decided_by = approver
        self._record(proposal, actor=approver)
        return proposal

    def reject(self, proposal_id: str, *, approver: str, reason: str = "") -> WriteProposal:
        proposal = self._require(proposal_id)
        proposal.status = WriteStatus.REJECTED
        proposal.decided_by = approver
        self._record(proposal, actor=approver, detail=reason)
        return proposal

    def execute(self, proposal_id: str) -> WriteProposal:
        proposal = self._require(proposal_id)
        if proposal.status is WriteStatus.EXECUTED:
            return proposal  # idempotent replay
        if proposal.status is not WriteStatus.APPROVED:
            raise PermissionError(
                f"refusing to execute {proposal.action}: status is "
                f"{proposal.status.value}, not approved"
            )

        action = self._actions[proposal.action]
        try:
            proposal.result = action.handler(
                tenant_id=proposal.tenant_id, **proposal.arguments
            )
        except Exception as exc:
            proposal.status = WriteStatus.FAILED
            proposal.error = f"{type(exc).__name__}: {exc}"
            self._record(proposal, actor="system", detail=proposal.error)
            raise
        proposal.status = WriteStatus.EXECUTED
        self._record(proposal, actor="system")
        return proposal

    def dry_run(self, name: str, **arguments: Any) -> str:
        action = self._actions.get(name)
        if action is None:
            raise KeyError(f"no write action named {name!r}")
        return action.describe(**arguments)

    def get(self, proposal_id: str) -> WriteProposal | None:
        return self._proposals.get(proposal_id)

    def pending(self) -> list[WriteProposal]:
        return [p for p in self._proposals.values() if p.status is WriteStatus.PROPOSED]

    def _require(self, proposal_id: str) -> WriteProposal:
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise KeyError(f"no such write proposal {proposal_id!r}")
        return proposal
