"""
model_registry.py

Responsibility:
    Register trained models and manage the "champion" alias in the
    MLflow Model Registry - SEPARATELY from training.

Why separate from training?
    Training produces a candidate. Deciding that a candidate should
    serve production traffic is a governance decision (review
    metrics, compare with the current champion, sign-off). Mixing
    the two would let any training run silently replace the
    production model.

Safety rules implemented here:
    - `register` creates a new version tagged "challenger" only.
    - `promote` refuses to replace an existing champion unless
      --replace-existing is passed explicitly.

Usage (from the project root):
    python -m src.model_registry register
    python -m src.model_registry promote --version 1
    python -m src.model_registry promote --version 2 --replace-existing
    python -m src.model_registry status
"""

import argparse
import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from src import config


class ChampionExistsError(RuntimeError):
    """Raised when promotion would silently replace a champion."""


def get_client(tracking_uri=config.DEFAULT_TRACKING_URI):
    # Explicit URI: never fall back to a remote server from env vars.
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient(tracking_uri=tracking_uri)


def read_latest_model_uri(model_dir=config.MODELS_DIR):
    """Read the MLflow model URI written by the last train.py run."""

    metadata_path = model_dir / config.METADATA_FILENAME

    if not metadata_path.exists():
        raise FileNotFoundError(
            f"{metadata_path} not found. Run 'python train.py' first "
            "or pass --model-uri."
        )

    return json.loads(metadata_path.read_text())["mlflow_model_uri"]


def get_champion_version(client, model_name):
    """Return the current champion version, or None if there is none."""

    try:
        return client.get_model_version_by_alias(
            model_name, config.CHAMPION_ALIAS
        )
    except MlflowException:
        # Raised when the alias or the registered model doesn't exist.
        return None


def register_model(
    client,
    model_uri,
    model_name=config.REGISTERED_MODEL_NAME,
):
    """
    Register a logged model as a new version and mark it challenger.

    Does NOT touch the champion alias.
    """

    version = mlflow.register_model(model_uri, model_name)

    client.set_registered_model_alias(
        model_name, config.CHALLENGER_ALIAS, version.version
    )

    return version


def promote_to_champion(
    client,
    version,
    model_name=config.REGISTERED_MODEL_NAME,
    replace_existing=False,
):
    """
    Point the champion alias at `version`.

    If a different champion already exists and replace_existing is
    False, raise instead of overwriting it.
    """

    current = get_champion_version(client, model_name)

    if (
        current is not None
        and str(current.version) != str(version)
        and not replace_existing
    ):
        raise ChampionExistsError(
            f"Version {current.version} is already the champion of "
            f"'{model_name}'. Review the metrics, then re-run with "
            "--replace-existing to replace it intentionally."
        )

    client.set_registered_model_alias(
        model_name, config.CHAMPION_ALIAS, str(version)
    )

    return current


def main():
    parser = argparse.ArgumentParser(
        description="Register churn models and manage the champion alias."
    )
    parser.add_argument("--tracking-uri", default=config.DEFAULT_TRACKING_URI)
    parser.add_argument("--model-name", default=config.REGISTERED_MODEL_NAME)
    sub = parser.add_subparsers(dest="command", required=True)

    reg = sub.add_parser("register", help="Register a trained model")
    reg.add_argument(
        "--model-uri", default=None,
        help="Default: URI from models/model_metadata.json",
    )

    promo = sub.add_parser("promote", help="Set the champion alias")
    promo.add_argument("--version", required=True)
    promo.add_argument("--replace-existing", action="store_true")

    sub.add_parser("status", help="Show champion and challenger")

    args = parser.parse_args()
    client = get_client(args.tracking_uri)

    if args.command == "register":
        model_uri = args.model_uri or read_latest_model_uri()
        version = register_model(client, model_uri, args.model_name)
        print(f"Registered '{args.model_name}' version {version.version} "
              f"as '{config.CHALLENGER_ALIAS}'. Champion unchanged.")

    elif args.command == "promote":
        try:
            previous = promote_to_champion(
                client, args.version, args.model_name,
                replace_existing=args.replace_existing,
            )
        except ChampionExistsError as error:
            raise SystemExit(f"Refused: {error}")
        previous_text = previous.version if previous else "none"
        print(f"Champion is now version {args.version} "
              f"(previous: {previous_text}).")

    elif args.command == "status":
        for alias in (config.CHAMPION_ALIAS, config.CHALLENGER_ALIAS):
            try:
                mv = client.get_model_version_by_alias(args.model_name, alias)
                print(f"{alias:<11} version {mv.version}  run {mv.run_id}")
            except MlflowException:
                print(f"{alias:<11} (none)")


if __name__ == "__main__":
    main()
