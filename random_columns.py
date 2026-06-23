import random

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
