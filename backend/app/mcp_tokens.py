"""Bearer tokens for the MCP endpoint, and a command line to manage them.

An AI agent can't complete the Authelia sign-in in a browser, so `/mcp` takes
a bearer token instead. Each token belongs to one reader and acts as that
reader: it reads their progress, drafts, and notes, and runs labs as them. A lab
that needs a GPU starts a paid RunPod job, the same as the **Run** button.

A token is `gpumcp_` followed by 32 random bytes in URL-safe Base64. Only its
SHA-256 is stored, along with its first 12 characters for display.

Run the command line inside the container:

    docker exec learn-gpu python -m backend.app.mcp_tokens create alice --name laptop
    docker exec learn-gpu python -m backend.app.mcp_tokens list
    docker exec learn-gpu python -m backend.app.mcp_tokens revoke 3
"""

from __future__ import annotations

import argparse
import hashlib
import secrets
import sys
import time
from typing import Any

from . import db

TOKEN_PREFIX = "gpumcp_"
PREFIX_LENGTH = 12
MAX_ACTIVE_TOKENS = 10
MAX_NAME_LENGTH = 64


class TokenLimitError(RuntimeError):
    """The reader already holds the maximum number of active tokens."""


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def create_token(username: str, name: str) -> dict[str, Any]:
    """Create a token for `username` and return its record with the secret.

    The secret is in the `token` key. It is not stored, so this is the only
    time anyone can read it.
    """
    name = name.strip()
    if not 1 <= len(name) <= MAX_NAME_LENGTH:
        raise ValueError(f"A token name must be 1 to {MAX_NAME_LENGTH} characters.")
    if db.count_tokens(username) >= MAX_ACTIVE_TOKENS:
        raise TokenLimitError(
            f"{username} already has {MAX_ACTIVE_TOKENS} active tokens. Revoke one first."
        )
    token = TOKEN_PREFIX + secrets.token_urlsafe(32)
    record = db.insert_token(username, name, hash_token(token), token[:PREFIX_LENGTH])
    record["token"] = token
    return record


def user_for_token(token: str) -> str | None:
    """The reader a presented token acts as, or None when it isn't valid.

    The lookup is by hash, so the comparison never touches the secret itself.
    """
    if not token.startswith(TOKEN_PREFIX):
        return None
    return db.token_user(hash_token(token))


# Command line -----------------------------------------------------------


def _when(timestamp: float | None) -> str:
    if timestamp is None:
        return "never"
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(timestamp))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.app.mcp_tokens",
        description="Create, list, and revoke bearer tokens for the /mcp endpoint.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="Create a token and print it once.")
    create.add_argument(
        "username",
        help="The reader the token acts as, as the app knows them after SSO_USERS mapping.",
    )
    create.add_argument("--name", default="cli", help="A label for the token. Default: cli.")

    listing = commands.add_parser("list", help="List active tokens.")
    listing.add_argument("username", nargs="?", help="Only this reader's tokens.")

    revoke = commands.add_parser("revoke", help="Revoke a token by its ID.")
    revoke.add_argument("id", type=int, help="The token ID that `list` prints.")

    args = parser.parse_args(argv)
    db.init_db()

    if args.command == "create":
        try:
            record = create_token(args.username, args.name)
        except (ValueError, TokenLimitError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(f"Created token {record['id']} ({record['name']}) for {args.username}.")
        print("Copy it now. It isn't shown again.")
        print(record["token"])
        return 0

    if args.command == "list":
        tokens = db.list_tokens(args.username)
        if not tokens:
            print("No active tokens.")
            return 0
        print(f"{'ID':>4}  {'USER':<16} {'NAME':<24} {'PREFIX':<12}  {'CREATED':<16}  LAST USED")
        for token in tokens:
            print(
                f"{token['id']:>4}  {token['username']:<16} {token['name']:<24} "
                f"{token['prefix']:<12}  {_when(token['created_at']):<16}  "
                f"{_when(token['last_used_at'])}"
            )
        return 0

    if not db.revoke_token(args.id):
        print(f"error: no active token with ID {args.id}.", file=sys.stderr)
        return 1
    print(f"Revoked token {args.id}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
