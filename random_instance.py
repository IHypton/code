import random
from itertools import product


def generate_random_instance(
    n_items=10,
    n_locations=3,
    min_pods_per_item=1,
    max_pods_per_item=3,
    q_min=1, q_max=10,
    d_min=1, d_max=20,
    M=1000,
    Q=None,
    seed=None,
):
    if seed is None:
        random.seed(seed)

    L = list(range(1, n_locations + 1))
    I = list(range(1, n_items + 1))

    if Q is None:
        Q = max(2, n_locations - 1)  # Standard: weniger Aufzüge als Standorte -> echte Auswahl nötig

    R_i = {}
    pod_counter = 0
    pod_names = []

    for i in I:
        n_pods = random.randint(min_pods_per_item, max_pods_per_item)
        pods_for_item = []
        for _ in range(n_pods):
            pod_name = f"P{pod_counter}"
            pods_for_item.append(pod_name)
            pod_names.append(pod_name)
            pod_counter += 1
        R_i[i] = pods_for_item

    R = pod_names

    # q_r: Workload pro Pod, unabhängig zufällig
    q = {r: random.randint(q_min, q_max) for r in R}

    # d_rl: Distanz pro Pod und Standort, unabhängig zufällig
    d = {}
    for r in R:
        for l in L:
            d[(r, l)] = random.randint(d_min, d_max)

    instance = {
        "I": I,
        "L": L,
        "R_i": R_i,
        "R": R,
        "q": q,
        "d": d,
        "Q": Q,
        "M": M,
    }
    return instance


def print_instance(instance):
    print("Items I:", instance["I"])
    print("Standorte L:", instance["L"])
    print("Q (max Aufzüge):", instance["Q"])
    print()
    print("R_i (Pods pro Item):")
    for i, pods in instance["R_i"].items():
        print(f"  Item {i}: {pods}")
    print()
    print("Workload q_r:")
    for r, val in instance["q"].items():
        print(f"  {r}: {val}")
    print()
    print("Distanz d_rl:")
    for (r, l), val in instance["d"].items():
        print(f"  ({r},{l}): {val}")


if __name__ == "__main__":
    instance = generate_random_instance(seed=42)
    print_instance(instance)


def generate_fully_random_instance(
    n_items_range=(5, 20),
    n_locations_range=(2, 6),
    pods_per_item_range=(1, 4),
    q_range_bounds=(1, 30),
    d_range_bounds=(1, 50),
    seed=None,
):
    rng = random.Random(seed)

    n_items = rng.randint(*n_items_range)
    n_locations = rng.randint(*n_locations_range)

    min_pods_per_item = rng.randint(pods_per_item_range[0], pods_per_item_range[1] - 1) \
        if pods_per_item_range[1] > pods_per_item_range[0] else pods_per_item_range[0]
    max_pods_per_item = rng.randint(min_pods_per_item, pods_per_item_range[1])

    # zufälliger Wertebereich für q_r 
    q_min = rng.randint(q_range_bounds[0], q_range_bounds[1] // 2)
    q_max = rng.randint(q_min + 1, q_range_bounds[1])

    # zufälliger Wertebereich für d_rl 
    d_min = rng.randint(d_range_bounds[0], d_range_bounds[1] // 2)
    d_max = rng.randint(d_min + 1, d_range_bounds[1])

    # Q zufällig
    Q = rng.randint(1, n_locations)

    # eigener Sub-Seed für die Werte-Generierung 
    sub_seed = rng.randint(0, 10**9)

    return generate_random_instance(
        n_items=n_items,
        n_locations=n_locations,
        min_pods_per_item=min_pods_per_item,
        max_pods_per_item=max_pods_per_item,
        q_min=q_min, q_max=q_max,
        d_min=d_min, d_max=d_max,
        Q=Q,
        seed=sub_seed,
    )


if __name__ == "__main__":
    print("\n\n=== Beispiel: voll-zufällige Instanz für ML-Daten ===\n")
    for i in range(3):
        inst = generate_fully_random_instance(seed=100 + i)
        print(f"--- Instanz {i} ---")
        print(f"n_items={len(inst['I'])}, n_locations={len(inst['L'])}, Q={inst['Q']}")
        print()