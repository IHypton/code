import random
from itertools import permutations

def sample_random_column(R_i, location, q, d, p_include=0.5, rng=None):
    if rng is None:
        rng = random

    pods = []
    items_covered = []

    for i, pods_i in R_i.items():
        if rng.random() < p_include:
            chosen_pod = rng.choice(pods_i)
            pods.append(chosen_pod)
            items_covered.append(i)

    if len(pods) == 0:
        # mind. 1 Pod, sonst Spalte verwerfen -> erzwinge mind. 1 Item
        i = rng.choice(list(R_i.keys()))
        chosen_pod = rng.choice(R_i[i])
        pods.append(chosen_pod)
        items_covered.append(i)

    workload = sum(q[r] for r in pods)
    distance = sum(d[(r, location)] for r in pods)

    return {
        "location": location,
        "pods": tuple(pods),
        "workload": workload,
        "distance": distance,
        "items": items_covered,
    }


def generate_random_columns(instance, n_columns_per_location=200, p_include=0.5, seed=None):
    """
    Erzeugt eine zufällige Stichprobe von Spalten je Standort.
    Duplikate (gleiche Pod-Menge + gleicher Standort) werden entfernt.
    """
    rng = random.Random(seed)

    R_i = instance["R_i"]
    L = instance["L"]
    q = instance["q"]
    d = instance["d"]

    seen = set()
    columns = []
    column_id = 0

    for l in L:
        attempts = 0
        n_generated = 0
        while n_generated < n_columns_per_location and attempts < n_columns_per_location * 10:
            attempts += 1
            col = sample_random_column(R_i, l, q, d, p_include=p_include, rng=rng)
            key = (l, col["pods"])
            if key in seen:
                continue
            seen.add(key)
            col["id"] = column_id
            columns.append(col)
            column_id += 1
            n_generated += 1

    return columns

def coverage_check(columns, I):
    """Prüft, ob alle Items durch mindestens eine generierte Spalte abgedeckt werden."""
    covered = set()
    for c in columns:
        covered.update(c["items"])
    missing = set(I) - covered
    return missing

def generate_lpt_heuristic_columns(instance):
    """Generiert Spalten basierend auf der LPT-Heuristik (Longest Processing Time)."""
    R_i = instance["R_i"]
    L = instance["L"]
    Q = instance["Q"]
    q = instance["q"]
    d = instance["d"]

    # Schritt 1: pro Item den Pod mit geringstem Workload wählen
    chosen_pod_per_item = {}
    for i, pods in R_i.items():
        best_pod = min(pods, key=lambda r: q[r])
        chosen_pod_per_item[i] = best_pod

    # Schritt 2: Items absteigend nach Workload des gewählten Pods sortieren
    items_sorted = sorted(
        chosen_pod_per_item.keys(),
        key=lambda i: q[chosen_pod_per_item[i]],
        reverse=True
    )

    # Schritt 3: LPT-Verteilung auf Q virtuelle Gruppen
    group_items = [[] for _ in range(Q)]
    group_load = [0] * Q

    for i in items_sorted:
        pod = chosen_pod_per_item[i]
        target = group_load.index(min(group_load))
        group_items[target].append((i, pod))
        group_load[target] += q[pod]

    # Schritt 4: für jede Gruppe den besten Standort finden (ohne Duplikate)
    non_empty_groups = [g for g in group_items if len(g) > 0]
    n_groups = len(non_empty_groups)

    best_assignment = None
    best_total_distance = None

    for loc_perm in permutations(L, n_groups):
        total_dist = 0
        for group, loc in zip(non_empty_groups, loc_perm):
            total_dist += sum(d[(pod, loc)] for (_, pod) in group)
        if best_total_distance is None or total_dist < best_total_distance:
            best_total_distance = total_dist
            best_assignment = loc_perm

    # Schritt 5: fertige Spalten bauen
    heuristic_columns = []
    for group, loc in zip(non_empty_groups, best_assignment):
        pods = tuple(pod for (_, pod) in group)
        items_covered = [i for (i, _) in group]
        workload = sum(q[r] for r in pods)
        distance = sum(d[(r, loc)] for r in pods)
        heuristic_columns.append({
            "location": loc,
            "pods": pods,
            "workload": workload,
            "distance": distance,
            "items": items_covered,
            "source": "lpt_heuristic",
        })

    return heuristic_columns


if __name__ == "__main__":
    from random_instance import generate_random_instance, print_instance

    instance = generate_random_instance(seed=42)
    print_instance(instance)

    columns = generate_random_columns(instance, n_columns_per_location=50, p_include=0.4, seed=1)
    print(f"\nAnzahl generierter (zufälliger) Spalten: {len(columns)}\n")

    for c in columns[:10]:
        print(f"ID={c['id']:>3} Loc={c['location']} Pods={c['pods']!s:<25} W={c['workload']:>3} D={c['distance']:>3} Items={c['items']}")

    missing = coverage_check(columns, instance["I"])
    print(f"\nNicht abgedeckte Items (sollte leer sein, sonst Resampling nötig): {missing}")

    print("\nLPT-Heuristik-Spalten:")
    lpt_cols = generate_lpt_heuristic_columns(instance)
    for c in lpt_cols:
        print(f"Loc={c['location']} Pods={c['pods']} W={c['workload']} D={c['distance']} Items={c['items']}")