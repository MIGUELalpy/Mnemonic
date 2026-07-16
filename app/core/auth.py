"""
Single-user bearer token authentication.

Every protected FastAPI route adds this as a dependency:

    @router.post("/submit", dependencies=[Depends(verify_token)])
    async def submit_code(...): ...

Or inject it to get the token string back:

    @router.get("/me")
    async def me(token: str = Depends(verify_token)): ...
"""

import hmac

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config import settings

# HTTPBearer extracts the token from the "Authorization: Bearer <token>" header.
# auto_error=False means it returns None instead of raising a 403 when the
# header is absent — we raise our own 401 below for a consistent error shape.
_bearer_scheme = HTTPBearer(auto_error=False)


async def verify_token(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> str:
    """
    FastAPI dependency that validates the static bearer token.

    Uses hmac.compare_digest for constant-time comparison to prevent
    timing-based token enumeration attacks, even on a local network.
    Raises:
        HTTPException 401 if the header is absent or the token is wrong.
    Returns:
        The validated token string (useful if the caller needs it).
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Expected: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # constant-time comparison — prevents timing attacks
    token_valid = hmac.compare_digest(
        credentials.credentials.encode(),
        settings.MNEMONIC_API_TOKEN.encode(),
    )

    if not token_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return credentials.credentials