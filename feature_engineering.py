FEATURE_COLS = [
    "n_items", "n_locations", "Q",
    "n_pods", "n_items_covered",
    "workload", "distance",
    "distance_per_item",
    "workload_rank_in_instance",
    "distance_rank_in_instance",
    "is_lpt_solution",
    "location_lpt_workload",
    "workload_vs_location_lpt",
    "max_lpt_workload",
    "beats_lpt_bottleneck",
]


def compute_features_for_columns(instance, columns, lpt_cols):
    """
    Berechnet für jede Spalte in `columns` ein Feature-Dict (ohne Label).
    `lpt_cols` wird genutzt, um die Kontext-Features relativ zur
    LPT-Heuristik-Lösung zu berechnen.
    Gibt eine Liste von Dicts zurück, parallel zu `columns` (gleiche Reihenfolge).
    """
    n_items = len(instance["I"])
    n_locations = len(instance["L"])

    workloads = [c["workload"] for c in columns]
    distances = [c["distance"] for c in columns]
    sorted_w = sorted(workloads)
    sorted_d = sorted(distances)

    lpt_location_workload = {c["location"]: c["workload"] for c in lpt_cols}
    max_lpt_workload = max((c["workload"] for c in lpt_cols), default=0)

    feature_rows = []
    for c in columns:
        n_pods = len(c["pods"])
        n_items_covered = len(c["items"])

        workload_rank = sorted_w.index(c["workload"]) / max(1, len(sorted_w) - 1)
        distance_rank = sorted_d.index(c["distance"]) / max(1, len(sorted_d) - 1)

        loc_lpt_workload = lpt_location_workload.get(c["location"], 0)
        workload_vs_location_lpt = c["workload"] - loc_lpt_workload
        beats_bottleneck = 1 if c["workload"] < max_lpt_workload else 0

        row = {
            "n_items": n_items,
            "n_locations": n_locations,
            "Q": instance["Q"],
            "n_pods": n_pods,
            "n_items_covered": n_items_covered,
            "workload": c["workload"],
            "distance": c["distance"],
            "distance_per_item": c["distance"] / n_items_covered if n_items_covered > 0 else 0,
            "workload_rank_in_instance": round(workload_rank, 4),
            "distance_rank_in_instance": round(distance_rank, 4),
            "is_lpt_solution": 1 if c.get("source") == "lpt_heuristic" else 0,
            "location_lpt_workload": loc_lpt_workload,
            "workload_vs_location_lpt": workload_vs_location_lpt,
            "max_lpt_workload": max_lpt_workload,
            "beats_lpt_bottleneck": beats_bottleneck,
        }
        feature_rows.append(row)

    return feature_rows