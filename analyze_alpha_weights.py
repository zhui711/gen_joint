"""Analyze polynomial timestep weights for anatomy-loss suppression."""

from statistics import mean


T_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ALPHAS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def format_row(values):
    return " | ".join(f"{value:>7.4f}" for value in values)


def main():
    print("Polynomial Timestep Weight Analysis")
    print("weight(t, alpha) = t ** alpha")
    print()

    header = ["alpha/t"] + [f"{t:.1f}" for t in T_VALUES]
    print(" | ".join(f"{item:>7}" for item in header))
    print("-" * (len(header) * 11))

    for alpha in ALPHAS:
        weights = [t ** alpha for t in T_VALUES]
        print(f"{alpha:>7.1f} | {format_row(weights)}")

    print()
    print("Low-t suppression summary (criterion: all t < 0.3 should have weight < 0.05)")
    low_t_values = [t for t in T_VALUES if t < 0.3]
    high_t_values = [t for t in T_VALUES if t > 0.7]
    candidate_alphas = []

    for alpha in ALPHAS:
        low_weights = [t ** alpha for t in low_t_values]
        high_weights = [t ** alpha for t in high_t_values]
        max_low = max(low_weights)
        min_high = min(high_weights)
        avg_high = mean(high_weights)
        satisfies_low = max_low < 0.05
        if satisfies_low:
            candidate_alphas.append(alpha)
        print(
            f"alpha={alpha:.1f}: max_low_t={max_low:.4f}, "
            f"min_high_t={min_high:.4f}, avg_high_t={avg_high:.4f}, "
            f"low_t_ok={satisfies_low}"
        )

    print()
    if candidate_alphas:
        joined = ", ".join(f"{alpha:.1f}" for alpha in candidate_alphas)
        print(f"Alphas meeting the low-t suppression criterion: {joined}")
    else:
        print("No alpha in the tested set satisfied the low-t suppression criterion.")


if __name__ == "__main__":
    main()
