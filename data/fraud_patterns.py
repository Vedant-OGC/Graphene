"""
Graphene — AML Fraud Pattern Planting

Plants 5 specific AML fraud patterns into the synthetic transaction dataset.
Each pattern is designed to be detectable by both graph queries and ML models.
These are the demo's "hero" detections.

Patterns implemented:
1. CIRCULAR_ROUND_TRIP — A→B→C→D→A within 72 hours
2. LAYERING — Large amount fans through intermediaries
3. STRUCTURING — Multiple transactions just under ₹49,999
4. DORMANT_ACTIVATION — 75+ day dormant account sudden spike
5. PROFILE_MISMATCH — Low-income profile with high-value credits
"""

import logging
import random
from datetime import datetime, timedelta
from uuid import uuid4

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

CONFIG = {
    "CIRCULAR_CYCLES": 4,
    "CIRCULAR_MIN_HOPS": 3,
    "CIRCULAR_MAX_HOPS": 5,
    "CIRCULAR_BASE_AMOUNT_MIN": 500_000,
    "CIRCULAR_BASE_AMOUNT_MAX": 2_500_000,
    "CIRCULAR_VARIANCE_PCT": 0.10,
    "CIRCULAR_TIME_WINDOW_HOURS": 72,
    "LAYERING_INSTANCES": 3,
    "LAYERING_MIN_SOURCE": 1_000_000,
    "LAYERING_MAX_SOURCE": 5_000_000,
    "LAYERING_INTERMEDIARIES_MIN": 4,
    "LAYERING_INTERMEDIARIES_MAX": 8,
    "LAYERING_FEE_PCT": 0.02,
    "LAYERING_TIME_WINDOW_HOURS": 48,
    "STRUCTURING_INSTANCES": 5,
    "STRUCTURING_THRESHOLD": 49_999,
    "STRUCTURING_TXN_MIN": 8,
    "STRUCTURING_TXN_MAX": 15,
    "STRUCTURING_TIME_WINDOW_HOURS": 24,
    "DORMANT_INSTANCES": 6,
    "DORMANT_DAYS_MIN": 75,
    "DORMANT_SPIKE_MULTIPLIER": 5.0,
    "DORMANT_FORWARD_PCT": 0.85,
    "MISMATCH_INSTANCES": 4,
    "MISMATCH_MIN_AMOUNT": 500_000,
    "MISMATCH_MAX_AMOUNT": 5_000_000,
    "MISMATCH_CREDITS_COUNT": 3,
}


def _random_timestamp_in_window(
    base: datetime, window_hours: int
) -> datetime:
    """Generate a random timestamp within a time window.

    Args:
        base: Start of the time window.
        window_hours: Window size in hours.

    Returns:
        A random datetime within [base, base + window_hours].
    """
    offset = timedelta(
        hours=random.uniform(0, window_hours)
    )
    return base + offset


def _pick_accounts(
    accounts_df: pd.DataFrame,
    n: int,
    exclude: set | None = None,
    account_type: str | None = None,
    category: str | None = None,
) -> list[str]:
    """Pick n distinct accounts matching optional filters.

    Args:
        accounts_df: DataFrame of all accounts.
        n: Number of accounts to pick.
        exclude: Account IDs to exclude.
        account_type: Filter by account type.
        category: Filter by customer category.

    Returns:
        List of account IDs.
    """
    mask = pd.Series(True, index=accounts_df.index)
    if account_type:
        mask &= accounts_df["account_type"] == account_type
    if category:
        mask &= accounts_df["customer_category"] == category
    if exclude:
        mask &= ~accounts_df["account_id"].isin(exclude)

    pool = accounts_df[mask]["account_id"].tolist()
    if len(pool) < n:
        fallback = accounts_df[
            ~accounts_df["account_id"].isin(exclude or set())
        ]["account_id"].tolist()
        pool = fallback

    return random.sample(pool, min(n, len(pool)))


def plant_circular_transactions(
    df: pd.DataFrame, accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Plant CIRCULAR (round-tripping) fraud patterns.

    Creates cycles A→B→C→D→A where total cycle completes within 72
    hours. Each leg has slightly different amounts (±5-15%) to evade
    exact matching.

    Args:
        df: Transactions DataFrame.
        accounts_df: Accounts DataFrame.

    Returns:
        Modified DataFrame with circular fraud transactions added.
    """
    n_cycles = CONFIG["CIRCULAR_CYCLES"]
    new_txns = []

    for cycle_idx in range(n_cycles):
        n_hops = random.randint(
            CONFIG["CIRCULAR_MIN_HOPS"],
            CONFIG["CIRCULAR_MAX_HOPS"],
        )
        cycle_accounts = _pick_accounts(
            accounts_df, n_hops, exclude=set()
        )
        base_amount = random.uniform(
            CONFIG["CIRCULAR_BASE_AMOUNT_MIN"],
            CONFIG["CIRCULAR_BASE_AMOUNT_MAX"],
        )
        base_time = datetime.now() - timedelta(
            days=random.randint(5, 60)
        )

        for hop in range(n_hops):
            sender = cycle_accounts[hop]
            receiver = cycle_accounts[(hop + 1) % n_hops]
            variance = random.uniform(
                -CONFIG["CIRCULAR_VARIANCE_PCT"],
                CONFIG["CIRCULAR_VARIANCE_PCT"],
            )
            amount = round(base_amount * (1 + variance), 2)
            timestamp = base_time + timedelta(
                hours=random.uniform(
                    hop * 4, (hop + 1) * 12
                )
            )

            new_txns.append({
                "txn_id": str(uuid4()),
                "timestamp": timestamp,
                "sender_account": sender,
                "receiver_account": receiver,
                "amount": amount,
                "txn_type": random.choice(
                    ["NEFT", "RTGS", "IMPS"]
                ),
                "channel": random.choice(
                    ["NETBANKING", "API", "MOBILE"]
                ),
                "status": "SUCCESS",
                "description": f"Transfer ref cycle-{cycle_idx}",
                "is_fraud": True,
                "fraud_type": "CIRCULAR_ROUND_TRIP",
            })

    fraud_df = pd.DataFrame(new_txns)
    result = pd.concat([df, fraud_df], ignore_index=True)
    logger.info(
        "Planted %d CIRCULAR_ROUND_TRIP instances "
        "(%d transactions)",
        n_cycles,
        len(new_txns),
    )
    return result


def plant_layering(
    df: pd.DataFrame, accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Plant LAYERING fraud patterns.

    One large source amount is split across 4-8 intermediate accounts,
    then recombined into a single destination within 48 hours. Each hop
    slightly reduces the amount (fee simulation).

    Args:
        df: Transactions DataFrame.
        accounts_df: Accounts DataFrame.

    Returns:
        Modified DataFrame with layering fraud transactions added.
    """
    n_instances = CONFIG["LAYERING_INSTANCES"]
    new_txns = []

    for inst_idx in range(n_instances):
        n_intermediaries = random.randint(
            CONFIG["LAYERING_INTERMEDIARIES_MIN"],
            CONFIG["LAYERING_INTERMEDIARIES_MAX"],
        )
        source = _pick_accounts(
            accounts_df, 1, account_type="CURRENT"
        )[0]
        dest = _pick_accounts(
            accounts_df, 1, exclude={source}
        )[0]
        intermediaries = _pick_accounts(
            accounts_df,
            n_intermediaries,
            exclude={source, dest},
        )
        source_amount = random.uniform(
            CONFIG["LAYERING_MIN_SOURCE"],
            CONFIG["LAYERING_MAX_SOURCE"],
        )
        base_time = datetime.now() - timedelta(
            days=random.randint(5, 60)
        )
        split_amount = source_amount / n_intermediaries
        fee_pct = CONFIG["LAYERING_FEE_PCT"]

        for i, mid_acc in enumerate(intermediaries):
            leg_amount = round(
                split_amount * (1 - fee_pct * random.uniform(0.5, 1.5)),
                2,
            )
            ts_fan_out = base_time + timedelta(
                hours=random.uniform(0, 6)
            )
            new_txns.append({
                "txn_id": str(uuid4()),
                "timestamp": ts_fan_out,
                "sender_account": source,
                "receiver_account": mid_acc,
                "amount": leg_amount,
                "txn_type": "NEFT",
                "channel": "NETBANKING",
                "status": "SUCCESS",
                "description": f"Payment ref layer-{inst_idx}-out",
                "is_fraud": True,
                "fraud_type": "LAYERING",
            })

            ts_fan_in = ts_fan_out + timedelta(
                hours=random.uniform(6, 36)
            )
            converge_amount = round(
                leg_amount * (1 - fee_pct * random.uniform(0.5, 1.0)),
                2,
            )
            new_txns.append({
                "txn_id": str(uuid4()),
                "timestamp": ts_fan_in,
                "sender_account": mid_acc,
                "receiver_account": dest,
                "amount": converge_amount,
                "txn_type": "IMPS",
                "channel": "API",
                "status": "SUCCESS",
                "description": f"Transfer ref layer-{inst_idx}-in",
                "is_fraud": True,
                "fraud_type": "LAYERING",
            })

    fraud_df = pd.DataFrame(new_txns)
    result = pd.concat([df, fraud_df], ignore_index=True)
    logger.info(
        "Planted %d LAYERING instances (%d transactions)",
        n_instances,
        len(new_txns),
    )
    return result


def plant_structuring(
    df: pd.DataFrame, accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Plant STRUCTURING (smurfing) fraud patterns.

    Multiple transactions from different accounts to the same
    destination, all just under ₹49,999 (below RBI reporting
    threshold of ₹50,000), all within 24 hours.

    Args:
        df: Transactions DataFrame.
        accounts_df: Accounts DataFrame.

    Returns:
        Modified DataFrame with structuring fraud transactions added.
    """
    n_instances = CONFIG["STRUCTURING_INSTANCES"]
    new_txns = []

    for inst_idx in range(n_instances):
        n_txns = random.randint(
            CONFIG["STRUCTURING_TXN_MIN"],
            CONFIG["STRUCTURING_TXN_MAX"],
        )
        dest = _pick_accounts(accounts_df, 1)[0]
        senders = _pick_accounts(
            accounts_df, n_txns, exclude={dest}
        )
        base_time = datetime.now() - timedelta(
            days=random.randint(5, 60)
        )

        for i, sender in enumerate(senders):
            amount = round(
                random.uniform(40_000, CONFIG["STRUCTURING_THRESHOLD"]),
                2,
            )
            timestamp = base_time + timedelta(
                hours=random.uniform(0, 20),
                minutes=random.randint(0, 59),
            )
            new_txns.append({
                "txn_id": str(uuid4()),
                "timestamp": timestamp,
                "sender_account": sender,
                "receiver_account": dest,
                "amount": amount,
                "txn_type": random.choice(["UPI", "IMPS", "NEFT"]),
                "channel": random.choice(["MOBILE", "NETBANKING"]),
                "status": "SUCCESS",
                "description": f"Payment ref struct-{inst_idx}",
                "is_fraud": True,
                "fraud_type": "STRUCTURING",
            })

    fraud_df = pd.DataFrame(new_txns)
    result = pd.concat([df, fraud_df], ignore_index=True)
    logger.info(
        "Planted %d STRUCTURING instances (%d transactions)",
        n_instances,
        len(new_txns),
    )
    return result


def plant_dormant_activation(
    df: pd.DataFrame, accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Plant DORMANT_ACTIVATION fraud patterns.

    Account with zero activity for 75+ days suddenly receives a
    transaction >5× their historical average, then immediately sends
    >80% of it onward.

    Args:
        df: Transactions DataFrame.
        accounts_df: Accounts DataFrame.

    Returns:
        Modified DataFrame with dormant activation fraud added.
    """
    n_instances = CONFIG["DORMANT_INSTANCES"]
    new_txns = []

    dormant_accounts = _pick_accounts(
        accounts_df, n_instances, account_type="DORMANT"
    )
    if len(dormant_accounts) < n_instances:
        extra = _pick_accounts(
            accounts_df,
            n_instances - len(dormant_accounts),
            exclude=set(dormant_accounts),
        )
        dormant_accounts.extend(extra)

    for inst_idx, dormant_acc in enumerate(dormant_accounts):
        incoming_amount = round(
            random.uniform(500_000, 5_000_000), 2
        )
        forward_amount = round(
            incoming_amount * CONFIG["DORMANT_FORWARD_PCT"]
            * random.uniform(0.95, 1.05),
            2,
        )
        source = _pick_accounts(
            accounts_df, 1, exclude={dormant_acc}
        )[0]
        dest = _pick_accounts(
            accounts_df, 1, exclude={dormant_acc, source}
        )[0]

        receive_time = datetime.now() - timedelta(
            days=random.randint(3, 15)
        )
        forward_time = receive_time + timedelta(
            hours=random.uniform(0.5, 8)
        )

        new_txns.append({
            "txn_id": str(uuid4()),
            "timestamp": receive_time,
            "sender_account": source,
            "receiver_account": dormant_acc,
            "amount": incoming_amount,
            "txn_type": "RTGS",
            "channel": "NETBANKING",
            "status": "SUCCESS",
            "description": f"Deposit ref dormant-{inst_idx}",
            "is_fraud": True,
            "fraud_type": "DORMANT_ACTIVATION",
        })

        new_txns.append({
            "txn_id": str(uuid4()),
            "timestamp": forward_time,
            "sender_account": dormant_acc,
            "receiver_account": dest,
            "amount": forward_amount,
            "txn_type": "NEFT",
            "channel": "NETBANKING",
            "status": "SUCCESS",
            "description": f"Transfer ref dormant-{inst_idx}",
            "is_fraud": True,
            "fraud_type": "DORMANT_ACTIVATION",
        })

    fraud_df = pd.DataFrame(new_txns)
    result = pd.concat([df, fraud_df], ignore_index=True)
    logger.info(
        "Planted %d DORMANT_ACTIVATION instances "
        "(%d transactions)",
        n_instances,
        len(new_txns),
    )
    return result


def plant_profile_mismatch(
    df: pd.DataFrame, accounts_df: pd.DataFrame
) -> pd.DataFrame:
    """Plant PROFILE_MISMATCH fraud patterns.

    Accounts declared as STUDENT or RETIRED category receive regular
    credits of 5-50 lakh (incompatible with their profile).

    Args:
        df: Transactions DataFrame.
        accounts_df: Accounts DataFrame.

    Returns:
        Modified DataFrame with profile mismatch fraud added.
    """
    n_instances = CONFIG["MISMATCH_INSTANCES"]
    new_txns = []

    student_accounts = _pick_accounts(
        accounts_df,
        n_instances // 2 + 1,
        category="STUDENT",
    )
    retired_accounts = _pick_accounts(
        accounts_df,
        n_instances // 2 + 1,
        category="RETIRED",
    )

    target_accounts = (
        student_accounts[: n_instances // 2]
        + retired_accounts[: n_instances - n_instances // 2]
    )

    for inst_idx, target_acc in enumerate(target_accounts):
        n_credits = CONFIG["MISMATCH_CREDITS_COUNT"]
        for credit_idx in range(n_credits):
            source = _pick_accounts(
                accounts_df, 1,
                exclude={target_acc},
                account_type="CURRENT",
            )[0]
            amount = round(
                random.uniform(
                    CONFIG["MISMATCH_MIN_AMOUNT"],
                    CONFIG["MISMATCH_MAX_AMOUNT"],
                ),
                2,
            )
            timestamp = datetime.now() - timedelta(
                days=random.randint(5, 60),
                hours=random.randint(0, 23),
            )

            new_txns.append({
                "txn_id": str(uuid4()),
                "timestamp": timestamp,
                "sender_account": source,
                "receiver_account": target_acc,
                "amount": amount,
                "txn_type": "NEFT",
                "channel": random.choice(
                    ["NETBANKING", "BRANCH"]
                ),
                "status": "SUCCESS",
                "description": (
                    f"Credit ref mismatch-{inst_idx}-{credit_idx}"
                ),
                "is_fraud": True,
                "fraud_type": "PROFILE_MISMATCH",
            })

    fraud_df = pd.DataFrame(new_txns)
    result = pd.concat([df, fraud_df], ignore_index=True)
    logger.info(
        "Planted %d PROFILE_MISMATCH instances "
        "(%d transactions)",
        n_instances,
        len(new_txns),
    )
    return result
