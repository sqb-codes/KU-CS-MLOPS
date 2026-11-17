"""Tests that registration and promotion are explicit and safe."""

import pytest

from src.model_registry import (
    ChampionExistsError,
    get_champion_version,
    get_client,
    promote_to_champion,
    register_model,
)

MODEL_NAME = "pytest-churn-model"


def test_existing_champion_is_not_replaced(training_run):

    client = get_client(training_run["tracking_uri"])
    model_uri = training_run["summary"]["mlflow_model_uri"]

    v1 = register_model(client, model_uri, MODEL_NAME)
    # Registering alone must not create a champion.
    assert get_champion_version(client, MODEL_NAME) is None

    promote_to_champion(client, v1.version, MODEL_NAME)

    v2 = register_model(client, model_uri, MODEL_NAME)

    # A second promotion without explicit consent is refused.
    with pytest.raises(ChampionExistsError):
        promote_to_champion(client, v2.version, MODEL_NAME)

    champion = get_champion_version(client, MODEL_NAME)
    assert str(champion.version) == str(v1.version)

    # With explicit consent, replacement succeeds.
    promote_to_champion(
        client, v2.version, MODEL_NAME, replace_existing=True
    )
    champion = get_champion_version(client, MODEL_NAME)
    assert str(champion.version) == str(v2.version)
