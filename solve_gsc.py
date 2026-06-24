import pulp
from random_instance import generate_random_instance, print_instance
from random_columns import generate_random_columns, generate_lpt_heuristic_columns, coverage_check


def solve_gsc(instance, columns, verbose=True):
    I = instance["I"]
    L = instance["L"]
    Q = instance["Q"]
    M = instance["M"]

    model = pulp.LpProblem("RMFSELP_GSC", pulp.LpMinimize)

    phi = pulp.LpVariable.dicts("phi", range(len(columns)), cat="Binary")
    b = pulp.LpVariable("b", lowBound=0)

    # Zielfunktion (8)
    model += M * b + pulp.lpSum(columns[k]["distance"] * phi[k] for k in range(len(columns)))

    # (9): jedes Item mind. 1x abgedeckt
    for i in I:
        relevant = [k for k in range(len(columns)) if i in columns[k]["items"]]
        if not relevant:
            raise ValueError(f"Item {i} wird von keiner generierten Spalte abgedeckt! Mehr Spalten sampeln.")
        model += pulp.lpSum(phi[k] for k in relevant) >= 1, f"cover_item_{i}"

    # (10): max Q Spalten insgesamt
    model += pulp.lpSum(phi[k] for k in range(len(columns))) <= Q, "max_elevators"

    # (11): höchstens 1 Aufzug pro Standort
    for l in L:
        model += pulp.lpSum(
            phi[k] for k in range(len(columns)) if columns[k]["location"] == l
        ) <= 1, f"one_elevator_per_loc_{l}"

    # (12): Workload-Bound pro Standort
    for l in L:
        model += pulp.lpSum(
            columns[k]["workload"] * phi[k]
            for k in range(len(columns)) if columns[k]["location"] == l
        ) <= b, f"workload_bound_{l}"

    model.solve(pulp.PULP_CBC_CMD(msg=0))

    status = pulp.LpStatus[model.status]
    result = {
        "status": status,
        "b": pulp.value(b) if status == "Optimal" else None,
        "objective": pulp.value(model.objective) if status == "Optimal" else None,
        "chosen_columns": [],
    }

    if status == "Optimal":
        for k in range(len(columns)):
            if pulp.value(phi[k]) > 0.5:
                result["chosen_columns"].append(columns[k])

    if verbose:
        print("Status:", status)
        print("Max Workload b =", result["b"])
        print("Zielfunktionswert G =", result["objective"])
        print()
        for c in result["chosen_columns"]:
            print(f"GEWÄHLT: ID={c['id']} Loc={c['location']} Pods={c['pods']} W={c['workload']} D={c['distance']} Items={c['items']}")

    return result


if __name__ == "__main__":
    instance = generate_random_instance(seed=42)
    print_instance(instance)

    random_cols = generate_random_columns(instance, n_columns_per_location=2000, p_include=0.5, seed=1)
    lpt_cols = generate_lpt_heuristic_columns(instance)


    columns = random_cols + lpt_cols
    for idx, c in enumerate(columns):
        c["id"] = idx

    missing = coverage_check(columns, instance["I"])
    print(f"\nNicht abgedeckte Items: {missing}")
    print(f"Anzahl Spalten gesamt: {len(columns)} (davon {len(lpt_cols)} LPT-Heuristik-Spalten)\n")

    result = solve_gsc(instance, columns)

    print("\n--- Quelle der gewählten Spalten ---")
    for c in result["chosen_columns"]:
        source = c.get("source", "random_sample")
        print(f"ID={c['id']} Loc={c['location']} Pods={c['pods']} W={c['workload']} D={c['distance']} -> Quelle: {source}")