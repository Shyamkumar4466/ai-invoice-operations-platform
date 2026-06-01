"""Password hashing and verification utilities."""

import bcrypt


_MAX_PASSWORD_BYTES = 72


def _encode_password(password: str) -> bytes:
    """Validate and encode a password for bcrypt."""
    if not isinstance(password, str):
        raise TypeError("Password must be a string.")
    if not password:
        raise ValueError("Password must not be empty.")

    try:
        encoded_password = password.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("Password contains invalid characters.") from exc

    if len(encoded_password) > _MAX_PASSWORD_BYTES:
        raise ValueError("Password must not exceed 72 bytes when UTF-8 encoded.")

    return encoded_password


def hash_password(password: str) -> str:
    """Return a salted bcrypt hash for a plain-text password."""
    encoded_password = _encode_password(password)

    try:
        return bcrypt.hashpw(encoded_password, bcrypt.gensalt()).decode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("Unable to hash password.") from exc


def verify_password(password: str, hashed_password: str) -> bool:
    """Return whether a plain-text password matches a stored bcrypt hash."""
    if not isinstance(hashed_password, str) or not hashed_password:
        return False

    try:
        encoded_password = _encode_password(password)
        encoded_hash = hashed_password.encode("utf-8")
        return bcrypt.checkpw(encoded_password, encoded_hash)
    except (TypeError, ValueError, UnicodeEncodeError):
        return False
