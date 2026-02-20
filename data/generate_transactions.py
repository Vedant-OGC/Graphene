"""
Graphene — Synthetic Transaction Data Generator

Generates realistic Indian banking transaction data for Graphene's demo.
Creates 500 accounts with distinct behavioral profiles and 5000 transactions
with realistic amounts, timestamps, and descriptions.

Usage:
    python data/generate_transactions.py
"""

import logging
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from faker import Faker

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.fraud_patterns import (
    plant_circular_transactions,
    plant_dormant_activation,
    plant_layering,
    plant_profile_mismatch,
    plant_structuring,
)

logging.basicConfig(
    level=logging.INFO,
    format="[GRAPHENE] %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

fake = Faker("en_IN")
Faker.seed(42)
np.random.seed(42)
random.seed(42)

CONFIG = {
    "N_ACCOUNTS": 500,
    "N_TRANSACTIONS": 5000,
    "FRAUD_RATIO": 0.08,
    "AMOUNT_MEAN": 50_000,
    "AMOUNT_STD": 200_000,
    "AMOUNT_MIN": 100,
    "AMOUNT_MAX": 50_000_000,
    "LOOKBACK_DAYS": 90,
    "SUCCESS_RATE": 0.95,
    "OUTPUT_DIR": str(Path(__file__).resolve().parent),
}

TXN_TYPES = ["NEFT", "RTGS", "IMPS", "UPI", "INTERNAL"]
TXN_TYPE_WEIGHTS = [0.20, 0.10, 0.25, 0.35, 0.10]

CHANNELS = ["MOBILE", "NETBANKING", "BRANCH", "ATM", "API"]
CHANNEL_WEIGHTS = [0.35, 0.25, 0.15, 0.10, 0.15]

ACCOUNT_TYPES = ["SAVINGS", "CURRENT", "SALARY", "DORMANT", "SHELL"]
ACCOUNT_TYPE_DIST = [0.40, 0.25, 0.20, 0.10, 0.05]

CUSTOMER_CATEGORIES = ["INDIVIDUAL", "BUSINESS", "STUDENT", "RETIRED"]
CATEGORY_DIST = [0.50, 0.25, 0.15, 0.10]

KYC_STATUSES = ["VERIFIED", "PENDING", "EXPIRED"]
KYC_DIST = [0.80, 0.12, 0.08]

BRANCH_CODES = [
    "UBIN0530001", "UBIN0530002", "UBIN0530003", "UBIN0530004",
    "UBIN0530005", "UBIN0530006", "UBIN0530007", "UBIN0530008",
    "UBIN0530009", "UBIN0530010", "UBIN0530011", "UBIN0530012",
    "UBIN0530013", "UBIN0530014", "UBIN0530015", "UBIN0530016",
    "UBIN0530017", "UBIN0530018", "UBIN0530019", "UBIN0530020",
]

DESCRIPTIONS = [
    "Salary credit", "Rent payment", "EMI payment", "Utility bill",
    "FD maturity credit", "Insurance premium", "Mutual fund SIP",
    "GST payment", "Vendor payment", "Supplier invoice",
    "Office rent", "Staff salary", "Freelance payment",
    "Consulting fee", "Commission credit", "Material purchase",
    "Travel reimbursement", "Medical expense", "Education fee",
    "Loan disbursement", "Dividend credit", "Share transfer",
    "Service charge", "Maintenance fee", "Subscription renewal",
    "Advertisement payment", "Legal fee", "Audit payment",
    "Deposit transfer", "Inter-branch transfer",
]


def generate_account_id() -> str:
    """Generate a unique account ID in ACC + 10-digit format."""
    return f"ACC{random.randint(1_000_000_000, 9_999_999_999)}"


def generate_accounts(n_accounts: int = 500) -> pd.DataFrame:
    """Generate realistic Indian bank account profiles.

    Creates accounts with distinct behavioral profiles:
    - SAVINGS: Individual low-value regular activity
    - CURRENT: Business high-value irregular flows
    - SALARY: Regular monthly credits 20k-80k
    - DORMANT: Zero activity 60+ days
    - SHELL: Pass-through accounts (>95% outflow)

    Args:
        n_accounts: Number of accounts to generate.

    Returns:
        DataFrame with account metadata.
    """
    logger.info("Generating %d account profiles...", n_accounts)

    accounts = []
    used_ids: set[str] = set()

    for _ in range(n_accounts):
        while True:
            acc_id = generate_account_id()
            if acc_id not in used_ids:
                used_ids.add(acc_id)
                break

        acc_type = np.random.choice(ACCOUNT_TYPES, p=ACCOUNT_TYPE_DIST)

        if acc_type == "CURRENT":
            category = "BUSINESS"
        elif acc_type == "SALARY":
            category = np.random.choice(
                ["INDIVIDUAL", "STUDENT"], p=[0.8, 0.2]
            )
        elif acc_type == "DORMANT":
            category = np.random.choice(
                ["INDIVIDUAL", "RETIRED"], p=[0.5, 0.5]
            )
        elif acc_type == "SHELL":
            category = np.random.choice(
                ["INDIVIDUAL", "BUSINESS"], p=[0.6, 0.4]
            )
        else:
            category = np.random.choice(
                CUSTOMER_CATEGORIES, p=CATEGORY_DIST
            )

        kyc = np.random.choice(KYC_STATUSES, p=KYC_DIST)
        age_days = max(
            30,
            int(np.random.lognormal(mean=6, sigma=1.2))
        )
        if acc_type == "DORMANT":
            age_days = max(365, age_days)

        accounts.append({
            "account_id": acc_id,
            "account_type": acc_type,
            "customer_name": fake.name(),
            "customer_category": category,
            "branch_code": random.choice(BRANCH_CODES),
            "kyc_status": kyc,
            "account_age_days": age_days,
            "risk_score": 0.0,
            "is_flagged": False,
        })

    df = pd.DataFrame(accounts)
    logger.info(
        "Accounts generated — Types: %s",
        df["account_type"].value_counts().to_dict(),
    )
    return df


def generate_weighted_timestamp(
    lookback_days: int = 90,
) -> datetime:
    """Generate a timestamp weighted toward business hours IST.

    Business hours (9am-6pm) get 70% of transactions.
    Remaining 30% spread across other hours.

    Args:
        lookback_days: Number of days to look back from now.

    Returns:
        A datetime object in IST.
    """
    now = datetime.now()
    start = now - timedelta(days=lookback_days)
    random_day = start + timedelta(
        days=random.uniform(0, lookback_days)
    )

    if random.random() < 0.70:
        hour = random.randint(9, 17)
    else:
        hour = random.choice(
            list(range(0, 9)) + list(range(18, 24))
        )

    minute = random.randint(0, 59)
    second = random.randint(0, 59)

    return random_day.replace(
        hour=hour, minute=minute, second=second, microsecond=0
    )


def generate_amount(
    account_type: str, txn_type: str
) -> float:
    """Generate realistic transaction amount based on profiles.

    Uses log-normal distribution with account type modifiers:
    - SALARY: 20k-80k range
    - CURRENT/BUSINESS: wider range up to 50L
    - SAVINGS: modest amounts
    - RTGS: minimum 2L (RBI rule)

    Args:
        account_type: The sending account's type.
        txn_type: The transaction type (NEFT, RTGS, etc).

    Returns:
        Transaction amount in INR.
    """
    if account_type == "SALARY":
        base = np.random.lognormal(mean=10.2, sigma=0.3)
        amount = np.clip(base, 20_000, 80_000)
    elif account_type in ("CURRENT", "SHELL"):
        base = np.random.lognormal(mean=11.5, sigma=1.5)
        amount = np.clip(
            base, CONFIG["AMOUNT_MIN"], CONFIG["AMOUNT_MAX"]
        )
    elif account_type == "DORMANT":
        base = np.random.lognormal(mean=9.5, sigma=0.8)
        amount = np.clip(base, 1_000, 500_000)
    else:
        base = np.random.lognormal(mean=10.0, sigma=1.0)
        amount = np.clip(
            base, CONFIG["AMOUNT_MIN"], 5_000_000
        )

    if txn_type == "RTGS":
        amount = max(amount, 200_000)
    elif txn_type == "UPI":
        amount = min(amount, 100_000)

    return round(float(amount), 2)


def generate_transaction_dataset(
    n_accounts: int = 500,
    n_transactions: int = 5000,
    fraud_ratio: float = 0.08,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate complete synthetic transaction dataset with fraud.

    Creates realistic Indian banking transactions with planted AML
    fraud patterns. Accounts have distinct behavioral profiles and
    transactions follow realistic distributions.

    Args:
        n_accounts: Number of account profiles to generate.
        n_transactions: Number of baseline transactions.
        fraud_ratio: Target fraction of fraudulent transactions.

    Returns:
        Tuple of (transactions_df, accounts_df).
    """
    accounts_df = generate_accounts(n_accounts)
    account_ids = accounts_df["account_id"].tolist()
    account_types = dict(
        zip(
            accounts_df["account_id"],
            accounts_df["account_type"],
        )
    )

    logger.info("Generating %d transactions...", n_transactions)
    transactions = []

    for _ in range(n_transactions):
        sender = random.choice(account_ids)
        receiver = random.choice(
            [a for a in account_ids if a != sender]
        )
        txn_type = np.random.choice(
            TXN_TYPES, p=TXN_TYPE_WEIGHTS
        )
        channel = np.random.choice(CHANNELS, p=CHANNEL_WEIGHTS)
        status = (
            "SUCCESS"
            if random.random() < CONFIG["SUCCESS_RATE"]
            else np.random.choice(["FAILED", "PENDING"])
        )
        timestamp = generate_weighted_timestamp(
            CONFIG["LOOKBACK_DAYS"]
        )
        amount = generate_amount(
            account_types.get(sender, "SAVINGS"), txn_type
        )
        description = random.choice(DESCRIPTIONS)

        transactions.append({
            "txn_id": str(uuid4()),
            "timestamp": timestamp,
            "sender_account": sender,
            "receiver_account": receiver,
            "amount": amount,
            "txn_type": txn_type,
            "channel": channel,
            "status": status,
            "description": description,
            "is_fraud": False,
            "fraud_type": None,
        })

    txn_df = pd.DataFrame(transactions)
    txn_df = txn_df.sort_values("timestamp").reset_index(drop=True)

    logger.info("Planting AML fraud patterns...")
    txn_df = plant_circular_transactions(txn_df, accounts_df)
    txn_df = plant_layering(txn_df, accounts_df)
    txn_df = plant_structuring(txn_df, accounts_df)
    txn_df = plant_dormant_activation(txn_df, accounts_df)
    txn_df = plant_profile_mismatch(txn_df, accounts_df)

    txn_df = txn_df.sort_values("timestamp").reset_index(drop=True)

    return txn_df, accounts_df


def save_datasets(
    txn_df: pd.DataFrame, accounts_df: pd.DataFrame
) -> None:
    """Save generated datasets to CSV files.

    Args:
        txn_df: Transactions DataFrame.
        accounts_df: Accounts DataFrame.
    """
    output_dir = CONFIG["OUTPUT_DIR"]
    os.makedirs(output_dir, exist_ok=True)

    txn_path = os.path.join(output_dir, "transactions.csv")
    acc_path = os.path.join(output_dir, "accounts.csv")

    txn_df.to_csv(txn_path, index=False)
    accounts_df.to_csv(acc_path, index=False)

    logger.info("Saved transactions to %s", txn_path)
    logger.info("Saved accounts to %s", acc_path)


def print_summary(
    txn_df: pd.DataFrame, accounts_df: pd.DataFrame
) -> None:
    """Print a summary of the generated dataset.

    Args:
        txn_df: Transactions DataFrame.
        accounts_df: Accounts DataFrame.
    """
    total = len(txn_df)
    fraud_count = txn_df["is_fraud"].sum()
    fraud_pct = (fraud_count / total) * 100 if total > 0 else 0

    print("\n" + "=" * 60)
    print("  GRAPHENE — Synthetic Data Generation Summary")
    print("=" * 60)
    print(f"  Total accounts:      {len(accounts_df):,}")
    print(f"  Total transactions:  {total:,}")
    print(f"  Fraud transactions:  {fraud_count:,} ({fraud_pct:.1f}%)")
    print(f"  Date range:          "
          f"{txn_df['timestamp'].min().strftime('%Y-%m-%d')} → "
          f"{txn_df['timestamp'].max().strftime('%Y-%m-%d')}")
    print()
    print("  Fraud Types Breakdown:")

    fraud_types = (
        txn_df[txn_df["is_fraud"]]
        .groupby("fraud_type")
        .size()
        .sort_values(ascending=False)
    )
    for ftype, count in fraud_types.items():
        print(f"    {ftype:30s} {count:5d}")

    print()
    print("  Account Types:")
    for atype, count in (
        accounts_df["account_type"].value_counts().items()
    ):
        print(f"    {atype:15s} {count:5d}")

    print("=" * 60 + "\n")


if __name__ == "__main__":
    txn_df, accounts_df = generate_transaction_dataset(
        n_accounts=CONFIG["N_ACCOUNTS"],
        n_transactions=CONFIG["N_TRANSACTIONS"],
        fraud_ratio=CONFIG["FRAUD_RATIO"],
    )
    save_datasets(txn_df, accounts_df)
    print_summary(txn_df, accounts_df)
