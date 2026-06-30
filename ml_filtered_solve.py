import time
import joblib
import pandas as pd
from random_instance import generate_fully_random_instance
from random_columns import generate_random_columns, generate_lpt_heuristic_columns, coverage_check
from solve_gsc import solve_gsc
from feature_engineering import compute_features_for_columns, FEATURE_COLS

MODEL_FILE = "model.pkl"
TOP_K = 200  # wie viele Zufallsspalten pro Instanz nach ML-Filter übrig bleiben
N_TEST_INSTANCES = 25
N_COLUMNS_PER_LOCATION_FULL = 2000  # Größe des "vollen" Pools zum Vergleich


def run_comparison(n_test_instances=N_TEST_INSTANCES, top_k=TOP_K, seed_offset=5000):
    """Führt den Vergleich zwischen dem vollen Spaltenpool und dem ML-gefilterten Pool durch."""
    print(f"Lade Modell aus {MODEL_FILE} ...")
    model = joblib.load(MODEL_FILE)
    print("Modell geladen. Starte Vergleich...\n")

    results = []

    for i in range(n_test_instances):
        print(f"[{i+1}/{n_test_instances}] Generiere Instanz...", flush=True)
        instance = generate_fully_random_instance(seed=seed_offset + i)
        print(f"    n_items={len(instance['I'])}, n_locations={len(instance['L'])}, Q={instance['Q']}", flush=True)

        random_cols = generate_random_columns(
            instance, n_columns_per_location=N_COLUMNS_PER_LOCATION_FULL,
            p_include=0.5, seed=seed_offset + 1000 + i
        )
        lpt_cols = generate_lpt_heuristic_columns(instance)
        print(f"    {len(random_cols)} Zufallsspalten + {len(lpt_cols)} LPT-Spalten generiert", flush=True)

        full_columns = random_cols + lpt_cols
        for idx, c in enumerate(full_columns):
            c["id"] = idx

        missing = coverage_check(full_columns, instance["I"])
        if missing:
            continue

        t0 = time.time()
        result_full = solve_gsc(instance, full_columns, verbose=False, time_limit=120)
        time_full = time.time() - t0
        print(f"    Voller Pool gelöst in {time_full:.2f}s, b={result_full['b']}, G={result_full['objective']}", flush=True)

        feature_rows = compute_features_for_columns(instance, random_cols, lpt_cols)
        X = pd.DataFrame(feature_rows)[FEATURE_COLS]
        scores = model.predict_proba(X)[:, 1]

        for c, score in zip(random_cols, scores):
            c["ml_score"] = score

        top_random_cols = sorted(random_cols, key=lambda c: c["ml_score"], reverse=True)[:top_k]

        filtered_columns = top_random_cols + lpt_cols
        for idx, c in enumerate(filtered_columns):
            c["id"] = idx

        missing_f = coverage_check(filtered_columns, instance["I"])
        if missing_f:
            # ML-Filter hat versehentlich ein Item komplett verworfen -> Fallback nötig
            results.append({
                "instance_id": seed_offset + i,
                "status": "filter_dropped_item_coverage",
            })
            continue

        t0 = time.time()
        result_filtered = solve_gsc(instance, filtered_columns, verbose=False)
        time_filtered = time.time() - t0
        print(f"    Gefilterter Pool gelöst in {time_filtered:.2f}s, b={result_filtered['b']}, G={result_filtered['objective']}", flush=True)

        # --- Diff-Analyse, falls Ergebnis abweicht ---
        if result_filtered["objective"] > result_full["objective"]:
            full_chosen_keys = {(c["location"], c["pods"]) for c in result_full["chosen_columns"]}
            filtered_chosen_keys = {(c["location"], c["pods"]) for c in result_filtered["chosen_columns"]}
            missing_keys = full_chosen_keys - filtered_chosen_keys

            print(f"    >>> ABWEICHUNG! Im Vollpool gewählte Spalte(n), die im Filter fehlen:")
            for key in missing_keys:
                loc, pods = key
                # Ist diese Spalte eine Zufallsspalte? Wo lag ihr ML-Score im Ranking?
                match = [c for c in random_cols if c["location"] == loc and c["pods"] == pods]
                if match:
                    c = match[0]
                    rank = sorted(random_cols, key=lambda x: x["ml_score"], reverse=True).index(c)
                    print(f"        Loc={loc} Pods={pods} W={c['workload']} D={c['distance']} "
                          f"ML-Score={c['ml_score']:.4f} -> Rang {rank} von {len(random_cols)} "
                          f"(Top-{top_k} Cutoff verpasst um {rank - top_k} Plätze)")
                else:
                    print(f"        Loc={loc} Pods={pods} (war eine LPT-Spalte, sollte eigentlich immer enthalten sein -> Bug?)")
        print()

        results.append({
            "instance_id": seed_offset + i,
            "n_columns_full": len(full_columns),
            "n_columns_filtered": len(filtered_columns),
            "objective_full": result_full["objective"],
            "objective_filtered": result_filtered["objective"],
            "b_full": result_full["b"],
            "b_filtered": result_filtered["b"],
            "time_full": time_full,
            "time_filtered": time_filtered,
            "status": "ok",
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    df = run_comparison()

    print(df.to_string(index=False))

    ok = df[df["status"] == "ok"]
    print(f"\n{len(ok)} von {len(df)} Instanzen erfolgreich verglichen "
          f"({len(df) - len(ok)} mit Coverage-Problem im gefilterten Pool)")

    if len(ok) > 0:
        print("\nZusammenfassung:")
        print(f"Ø Zielfunktionswert voller Pool:      {ok['objective_full'].mean():.1f}")
        print(f"Ø Zielfunktionswert ML-gefiltert:      {ok['objective_filtered'].mean():.1f}")
        n_identical = (ok['objective_full'] == ok['objective_filtered']).sum()
        n_worse = (ok['objective_filtered'] > ok['objective_full']).sum()
        n_better = (ok['objective_filtered'] < ok['objective_full']).sum()
        print(f"Identisches Ergebnis: {n_identical}, ML-gefiltert schlechter: {n_worse}, "
              f"ML-gefiltert besser: {n_better}")

        print(f"\nØ Solver-Zeit voller Pool:      {ok['time_full'].mean():.4f}s")
        print(f"Ø Solver-Zeit ML-gefiltert:      {ok['time_filtered'].mean():.4f}s")
        speedup = ok['time_full'].mean() / max(ok['time_filtered'].mean(), 1e-9)
        print(f"Speedup: {speedup:.2f}x")