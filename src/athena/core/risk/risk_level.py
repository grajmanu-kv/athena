from __future__ import annotations

from athena import ActionClass, ConsequenceClass, RiskLevel

_MATRIX: dict[tuple[ActionClass, ConsequenceClass], RiskLevel] = {
    # inform
    (ActionClass.inform, ConsequenceClass.negligible):  RiskLevel.low,
    (ActionClass.inform, ConsequenceClass.low):         RiskLevel.low,
    (ActionClass.inform, ConsequenceClass.significant): RiskLevel.medium,
    (ActionClass.inform, ConsequenceClass.critical):    RiskLevel.medium,

    # recommend
    (ActionClass.recommend, ConsequenceClass.negligible):  RiskLevel.low,
    (ActionClass.recommend, ConsequenceClass.low):         RiskLevel.medium,
    (ActionClass.recommend, ConsequenceClass.significant): RiskLevel.medium,
    (ActionClass.recommend, ConsequenceClass.critical):    RiskLevel.high,

    # decide
    (ActionClass.decide, ConsequenceClass.negligible):  RiskLevel.medium,
    (ActionClass.decide, ConsequenceClass.low):         RiskLevel.medium,
    (ActionClass.decide, ConsequenceClass.significant): RiskLevel.high,
    (ActionClass.decide, ConsequenceClass.critical):    RiskLevel.high,

    # automate
    (ActionClass.automate, ConsequenceClass.negligible):  RiskLevel.medium,
    (ActionClass.automate, ConsequenceClass.low):         RiskLevel.high,
    (ActionClass.automate, ConsequenceClass.significant): RiskLevel.high,
    (ActionClass.automate, ConsequenceClass.critical):    RiskLevel.critical,
}


def derive_risk_level(
    action_class: ActionClass,
    consequence_class: ConsequenceClass,
) -> RiskLevel:
    """
    Derive RiskLevel from ActionClass × ConsequenceClass.

    Raises
    ------
    KeyError
        If the combination is not in the matrix — should not happen
        as the matrix covers all 16 valid combinations.
    """
    key = (action_class, consequence_class)
    risk = _MATRIX.get(key)
    if risk is None:
        raise KeyError(
            f"No risk level defined for "
            f"({action_class}, {consequence_class}). "
            f"This is a library bug — all 16 combinations must be covered."
        )
    return risk
