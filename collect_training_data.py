import csv
import time
from random_instance import generate_fully_random_instance
from random_columns import generate_random_columns, generate_lpt_heuristic_columns, coverage_check
from solve_gsc import solve_gsc

N_INSTANCES = 1000
OUTPUT_CSV = "training_data_large.csv"

FIELDNAMES = [
    "instance_id",
    "n_items", "n_locations", "Q",
    "location",
    "n_pods", "n_items_covered",
    "workload", "distance",
    "distance_per_item",
    "workload_rank_in_instance",   # 0 = niedrigster Workload aller Spalten dieser Instanz
    "distance_rank_in_instance",
    "is_lpt_solution",
    "chosen",   # LABEL: 1 wenn vom Solver gewählt, sonst 0
]

def build_feature_rows(instance_id, instance, columns, chosen_ids):
    """Erzeugt für jede Spalte eine Feature-Zeile inkl. Label."""
    n_items = len(instance["I"])
    n_locations = len(instance["L"])

    workloads = [c["workload"] for c in columns]
    distances = [c["distance"] for c in columns]

    # einfache Rang-Features (0 = kleinster Wert in dieser Instanz)
    sorted_w = sorted(workloads)
    sorted_d = sorted(distances)

    rows = []
    for c in columns:
        n_pods = len(c["pods"])
        n_items_covered = len(c["items"])

        workload_rank = sorted_w.index(c["workload"]) / max(1, len(sorted_w) - 1)
        distance_rank = sorted_d.index(c["distance"]) / max(1, len(sorted_d) - 1)

        row = {
            "instance_id": instance_id,
            "n_items": n_items,
            "n_locations": n_locations,
            "Q": instance["Q"],
            "location": c["location"],
            "n_pods": n_pods,
            "n_items_covered": n_items_covered,
            "workload": c["workload"],
            "distance": c["distance"],
            "distance_per_item": c["distance"] / n_items_covered if n_items_covered > 0 else 0,
            "workload_rank_in_instance": round(workload_rank, 4),
            "distance_rank_in_instance": round(distance_rank, 4),
            "is_lpt_solution": 1 if c.get("source") == "lpt_heuristic" else 0,
            "chosen": 1 if c["id"] in chosen_ids else 0,
        }
        rows.append(row)
    return rows


def collect_training_data(n_instances=N_INSTANCES, output_csv=OUTPUT_CSV):
    all_rows = []
    n_optimal = 0
    n_failed = 0
    start = time.time()

    for inst_id in range(n_instances):
        try:
            instance = generate_fully_random_instance(seed=1000 + inst_id)

            random_cols = generate_random_columns(
                instance, n_columns_per_location=300, p_include=0.5, seed=2000 + inst_id
            )
            lpt_cols = generate_lpt_heuristic_columns(instance)

            columns = random_cols + lpt_cols
            for idx, c in enumerate(columns):
                c["id"] = idx

            missing = coverage_check(columns, instance["I"])
            if missing:
                # sollte durch LPT-Spalten eigentlich nie passieren, aber sicherheitshalber
                n_failed += 1
                continue

            result = solve_gsc(instance, columns, verbose=False)

            if result["status"] != "Optimal":
                n_failed += 1
                continue

            n_optimal += 1
            chosen_ids = {c["id"] for c in result["chosen_columns"]}

            rows = build_feature_rows(inst_id, instance, columns, chosen_ids)
            all_rows.extend(rows)

        except Exception as e:
            n_failed += 1
            print(f"  [Instanz {inst_id}] Fehler: {e}")

        if (inst_id + 1) % 25 == 0:
            elapsed = time.time() - start
            print(f"  {inst_id + 1}/{n_instances} Instanzen verarbeitet "
                  f"({elapsed:.1f}s, {n_optimal} optimal, {n_failed} fehlgeschlagen)")

    # CSV schreiben
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nFertig: {n_optimal} optimale Instanzen, {n_failed} fehlgeschlagen.")
    print(f"Insgesamt {len(all_rows)} Spalten-Zeilen (Trainingsdaten) -> {output_csv}")

    n_chosen = sum(r["chosen"] for r in all_rows)
    print(f"Davon als 'gewählt' (chosen=1) gelabelt: {n_chosen} ({100*n_chosen/max(1,len(all_rows)):.1f}%)")


if __name__ == "__main__":
    collect_training_data()
