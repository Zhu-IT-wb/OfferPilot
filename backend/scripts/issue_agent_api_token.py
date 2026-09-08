import argparse

from app.core.config import settings
from app.services.agent_api_auth import issue_agent_api_actor_token


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Issue a short-lived Bearer token for the OfferPilot Agent API."
    )
    parser.add_argument("--user-id", required=True)
    parser.add_argument(
        "--ttl-seconds",
        type=int,
        default=settings.agent_api_token_ttl_seconds,
    )
    args = parser.parse_args()
    print(
        issue_agent_api_actor_token(
            settings.agent_api_signing_secret,
            args.user_id,
            ttl_seconds=args.ttl_seconds,
        )
    )


if __name__ == "__main__":
    main()
