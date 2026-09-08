import base64
import hashlib
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, Protocol

from cryptography.fernet import Fernet, InvalidToken


class FeishuCredentialError(RuntimeError):
    pass


class FeishuCredentialDecryptionError(FeishuCredentialError):
    pass


@dataclass(frozen=True)
class FeishuTokenBundle:
    access_token: str = field(repr=False)
    refresh_token: Optional[str] = field(default=None, repr=False)
    access_expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    refresh_expires_at: Optional[datetime] = None
    scope: str = ""
    token_type: str = "Bearer"

    def __post_init__(self) -> None:
        if not self.access_token:
            raise ValueError("Feishu access_token must not be empty.")
        _require_aware(self.access_expires_at, "access_expires_at")
        if self.refresh_expires_at is not None:
            _require_aware(self.refresh_expires_at, "refresh_expires_at")


@dataclass(frozen=True)
class FeishuUserCredential:
    owner_id: str
    open_id: str
    access_token: str = field(repr=False)
    refresh_token: Optional[str] = field(default=None, repr=False)
    access_expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(hours=1)
    )
    refresh_expires_at: Optional[datetime] = None
    scope: str = ""
    token_type: str = "Bearer"
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self) -> None:
        if not self.owner_id or not self.open_id or not self.access_token:
            raise ValueError("Feishu user credential is incomplete.")
        _require_aware(self.access_expires_at, "access_expires_at")
        _require_aware(self.updated_at, "updated_at")
        if self.refresh_expires_at is not None:
            _require_aware(self.refresh_expires_at, "refresh_expires_at")

    @classmethod
    def from_token_bundle(
        cls,
        owner_id: str,
        open_id: str,
        bundle: FeishuTokenBundle,
        updated_at: Optional[datetime] = None,
    ) -> "FeishuUserCredential":
        return cls(
            owner_id=owner_id,
            open_id=open_id,
            access_token=bundle.access_token,
            refresh_token=bundle.refresh_token,
            access_expires_at=bundle.access_expires_at,
            refresh_expires_at=bundle.refresh_expires_at,
            scope=bundle.scope,
            token_type=bundle.token_type,
            updated_at=updated_at or datetime.now(timezone.utc),
        )


class FeishuUserCredentialRepository(Protocol):
    def save(self, credential: FeishuUserCredential) -> FeishuUserCredential: ...

    def get(self, owner_id: str) -> Optional[FeishuUserCredential]: ...

    def delete(self, owner_id: str) -> bool: ...


class SQLiteFeishuUserCredentialRepository:
    def __init__(self, database_path: str, encryption_secret: str) -> None:
        if not encryption_secret:
            raise FeishuCredentialError(
                "Feishu user credential encryption secret is not configured."
            )
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._cipher = Fernet(
            base64.urlsafe_b64encode(
                hashlib.sha256(encryption_secret.encode("utf-8")).digest()
            )
        )
        self._initialize()

    def save(self, credential: FeishuUserCredential) -> FeishuUserCredential:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO feishu_user_credentials (
                    owner_id, open_id, access_token_ciphertext,
                    refresh_token_ciphertext, access_expires_at,
                    refresh_expires_at, scope, token_type, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(owner_id) DO UPDATE SET
                    open_id = excluded.open_id,
                    access_token_ciphertext = excluded.access_token_ciphertext,
                    refresh_token_ciphertext = excluded.refresh_token_ciphertext,
                    access_expires_at = excluded.access_expires_at,
                    refresh_expires_at = excluded.refresh_expires_at,
                    scope = excluded.scope,
                    token_type = excluded.token_type,
                    updated_at = excluded.updated_at
                """,
                (
                    credential.owner_id,
                    credential.open_id,
                    self._encrypt(credential.access_token),
                    self._encrypt(credential.refresh_token)
                    if credential.refresh_token
                    else None,
                    credential.access_expires_at.astimezone(timezone.utc).isoformat(),
                    (
                        credential.refresh_expires_at.astimezone(timezone.utc).isoformat()
                        if credential.refresh_expires_at
                        else None
                    ),
                    credential.scope,
                    credential.token_type,
                    credential.updated_at.astimezone(timezone.utc).isoformat(),
                ),
            )
            connection.commit()
        return credential

    def get(self, owner_id: str) -> Optional[FeishuUserCredential]:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT owner_id, open_id, access_token_ciphertext,
                       refresh_token_ciphertext, access_expires_at,
                       refresh_expires_at, scope, token_type, updated_at
                FROM feishu_user_credentials
                WHERE owner_id = ?
                """,
                (owner_id,),
            ).fetchone()
        if row is None:
            return None
        return FeishuUserCredential(
            owner_id=row["owner_id"],
            open_id=row["open_id"],
            access_token=self._decrypt(row["access_token_ciphertext"]),
            refresh_token=(
                self._decrypt(row["refresh_token_ciphertext"])
                if row["refresh_token_ciphertext"]
                else None
            ),
            access_expires_at=datetime.fromisoformat(row["access_expires_at"]),
            refresh_expires_at=(
                datetime.fromisoformat(row["refresh_expires_at"])
                if row["refresh_expires_at"]
                else None
            ),
            scope=row["scope"] or "",
            token_type=row["token_type"] or "Bearer",
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def delete(self, owner_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM feishu_user_credentials WHERE owner_id = ?",
                (owner_id,),
            )
            connection.commit()
        return cursor.rowcount > 0

    def _encrypt(self, value: str) -> str:
        return self._cipher.encrypt(value.encode("utf-8")).decode("ascii")

    def _decrypt(self, value: str) -> str:
        try:
            return self._cipher.decrypt(value.encode("ascii")).decode("utf-8")
        except (InvalidToken, UnicodeDecodeError, ValueError) as exc:
            raise FeishuCredentialDecryptionError(
                "Stored Feishu credential cannot be decrypted with the configured secret."
            ) from exc

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS feishu_user_credentials (
                    owner_id TEXT PRIMARY KEY,
                    open_id TEXT NOT NULL,
                    access_token_ciphertext TEXT NOT NULL,
                    refresh_token_ciphertext TEXT,
                    access_expires_at TEXT NOT NULL,
                    refresh_expires_at TEXT,
                    scope TEXT NOT NULL DEFAULT '',
                    token_type TEXT NOT NULL DEFAULT 'Bearer',
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection


def _require_aware(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware.")
