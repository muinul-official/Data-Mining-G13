import pandas as pd
import numpy as np

def simulate_ecommerce_dataset(n_samples=1800, random_seed=42):
    """
    Simulates a realistic e-commerce customer churn dataset.
    
    Args:
        n_samples (int): Number of rows to generate.
        random_seed (int): Random seed for reproducibility.
        
    Returns:
        pd.DataFrame: Simulated dataset.
    """
    rng = np.random.default_rng(random_seed)
    
    def clip(a, lo, hi):
        return np.clip(a, lo, hi)

    states = ["Selangor", "KL", "Johor", "Penang", "Perak", "Sabah", "Sarawak", "Pahang", "Kelantan", "Terengganu"]
    income_bands = ["B40", "M40", "T20"]
    segments = ["new", "regular", "loyal", "premium"]
    genders = ["F", "M"]

    df = pd.DataFrame({
        "customer_id": [f"C{100000+i}" for i in range(n_samples)],
        "age": clip(rng.normal(30, 9, n_samples).round(), 18, 65).astype(int),
        "gender": rng.choice(genders, n_samples, p=[0.55, 0.45]),
        "state": rng.choice(states, n_samples, p=[0.22, 0.14, 0.12, 0.08, 0.08, 0.07, 0.07, 0.08, 0.07, 0.07]),
        "income_band": rng.choice(income_bands, n_samples, p=[0.48, 0.40, 0.12]),
        "tenure_days": clip(rng.gamma(shape=2.0, scale=120, size=n_samples).round(), 7, 1500).astype(int),
        "segment": rng.choice(segments, n_samples, p=[0.25, 0.45, 0.22, 0.08]),
    })

    # Behavioural multipliers
    seg_mult_orders = df["segment"].map({"new": 0.6, "regular": 1.0, "loyal": 1.5, "premium": 1.8}).values
    seg_mult_aov = df["segment"].map({"new": 0.9, "regular": 1.0, "loyal": 1.1, "premium": 1.3}).values

    # Orders & Spend
    df["orders_90d"] = clip(rng.poisson(lam=3.5 * seg_mult_orders), 0, 40).astype(int)
    df["orders_30d"] = clip((df["orders_90d"] * rng.uniform(0.15, 0.65, n_samples)).round(), 0, 20).astype(int)
    df["avg_order_value"] = clip(rng.lognormal(mean=3.7, sigma=0.35, size=n_samples) * seg_mult_aov, 15, 600).round(2)
    df["total_spend_90d"] = (df["orders_90d"] * df["avg_order_value"] * rng.uniform(0.8, 1.2, n_samples)).round(2)

    # Activity & Engagement
    df["days_since_last_order"] = clip(rng.integers(0, 120, n_samples) - (df["orders_30d"] * rng.integers(0, 3, n_samples)), 0, 120).astype(int)
    df["browse_sessions_30d"] = clip(rng.poisson(lam=10 * seg_mult_orders) + rng.integers(0, 10, n_samples), 0, 120).astype(int)
    
    # Proportion features
    df["cart_abandon_rate"] = clip(rng.beta(a=2.0, b=5.0, size=n_samples) + (df["days_since_last_order"] / 300), 0, 0.98).round(3)
    df["return_rate"] = clip(rng.beta(a=1.3, b=15, size=n_samples) + rng.normal(0, 0.01, n_samples), 0, 0.60).round(3)
    df["refund_count_90d"] = clip(rng.poisson(lam=df["orders_90d"] * df["return_rate"] * 0.7), 0, 20).astype(int)
    
    # Marketing & Service
    df["discount_ratio"] = clip(rng.beta(a=2.2, b=4.5, size=n_samples) + (df["income_band"].map({"B40": 0.08, "M40": 0.03, "T20": -0.02}).values), 0, 0.95).round(3)
    df["coupon_used_rate"] = clip(rng.beta(a=2.0, b=3.0, size=n_samples) + (df["discount_ratio"] - 0.3), 0, 1.0).round(3)
    df["delivery_delay_rate"] = clip(rng.beta(a=1.5, b=12.0, size=n_samples) + rng.normal(0, 0.02, n_samples), 0, 0.70).round(3)
    df["late_delivery_count"] = clip(rng.poisson(lam=df["orders_90d"] * df["delivery_delay_rate"] * 0.6), 0, 15).astype(int)
    df["failed_delivery_count"] = clip(rng.poisson(lam=df["orders_90d"] * 0.01), 0, 5).astype(int)
    df["tickets_90d"] = clip(rng.poisson(lam=0.3 + (df["late_delivery_count"] * 0.15) + (df["refund_count_90d"] * 0.12)), 0, 12).astype(int)
    df["avg_first_response_hours"] = clip(rng.lognormal(mean=2.0, sigma=0.5, size=n_samples) + df["tickets_90d"] * 0.8, 0.5, 72).round(2)
    
    # Platform Engagement
    df["email_open_rate"] = clip(rng.beta(a=2.5, b=3.5, size=n_samples) - (df["days_since_last_order"] / 250), 0, 1.0).round(3)
    df["push_open_rate"] = clip(rng.beta(a=2.0, b=4.0, size=n_samples) - (df["days_since_last_order"] / 300), 0, 1.0).round(3)
    df["loyalty_points_balance"] = clip((df["total_spend_90d"] * rng.uniform(0.3, 1.2, n_samples)).round(), 0, 50000).astype(int)
    df["app_rating"] = clip(rng.normal(4.1, 0.7, n_samples) - (df["late_delivery_count"] * 0.08) - (df["tickets_90d"] * 0.06), 1, 5).round(1)

    # Churn Label Sigma
    def sigmoid(x): return 1 / (1 + np.exp(-x))
    risk = (
        0.035 * df["days_since_last_order"]
        + 0.9 * df["delivery_delay_rate"] * 10
        + 0.45 * df["refund_count_90d"]
        + 0.35 * df["tickets_90d"]
        + 0.25 * df["cart_abandon_rate"] * 10
        - 0.30 * df["orders_30d"]
        - 0.18 * df["email_open_rate"] * 10
        - 0.15 * df["push_open_rate"] * 10
        - 0.00002 * df["loyalty_points_balance"]
        - 0.40 * (df["app_rating"] - 3.0)
    )
    p = sigmoid((risk - np.percentile(risk, 55)) / 2.2)
    df["churn"] = (rng.uniform(0, 1, n_samples) < p).astype(int)

    return df
