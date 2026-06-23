import pulp
from itertools import product

# ==================================================
# SCHRITT 1: INSTANZ
# ==================================================
# Notation exakt nach Paper (Tabelle 1 / Abschnitt 3.1):
#
# I        = Menge der Artikel (Items)
# R_i      = Menge der Pods, von denen Artikel i gepickt werden kann
# R        = Menge aller Pods (Vereinigung aller R_i)
# L        = Menge der potenziellen Aufzugsstandorte
# Q        = maximale Anzahl Aufzüge
# d[r,l]   = Distanz für Pod r via Standort l
# q[r]     = Workload (Aufzugsbelegung) von Pod r
# M        = Gewichtungsfaktor (big number) für lexikografische Priorität

Q = 2

L = [1, 2]  # Standorte

# R_i: Item -> Liste der Pods, die dieses Item liefern können
R_i = {
    1: ["A", "B"],
    2: ["C"],
    3: ["D"],
}

I = list(R_i.keys())

# R = Vereinigung aller R_i (Paper: R = ⋃_{i∈I} R_i)
R = [pod for pods in R_i.values() for pod in pods]

# q_r: Workload pro Pod
q = {
    "A": 3,
    "B": 4,
    "C": 5,
    "D": 2,
}

# d_{r,l}: Distanz Pod r -> Standort l
d = {
    ("A", 1): 8,  ("A", 2): 6,
    ("B", 1): 10, ("B", 2): 5,
    ("C", 1): 12, ("C", 2): 9,
    ("D", 1): 7,  ("D", 2): 11,
}

M = 1000  # big number für lexikografische Gewichtung

# ==================================================
# SCHRITT 2: SPALTENGENERIERUNG
# ==================================================

def generate_valid_pod_sets(R_i):
    
    choices_per_item = [[None] + pods for pods in R_i.values()]

    for combo in product(*choices_per_item):
        pods = tuple(p for p in combo if p is not None)
        if len(pods) > 0:
            yield pods


columns = []
column_id = 0

for l in L:
    for pods in generate_valid_pod_sets(R_i):

        workload = sum(q[r] for r in pods)
        distance = sum(d[(r, l)] for r in pods)

        # welche Items werden durch diese Spalte gedeckt?
        covered_items = [
            i for i, pods_i in R_i.items()
            if any(r in pods for r in pods_i)
        ]

        columns.append({
            "id": column_id,
            "location": l,
            "pods": pods,
            "workload": workload,
            "distance": distance,
            "items": covered_items,
        })
        column_id += 1

print(f"Anzahl generierter Spalten: {len(columns)}\n")

for c in columns:
    print(
        f"ID={c['id']:>2} Loc={c['location']} "
        f"Pods={c['pods']!s:<15} "
        f"W={c['workload']:>2} D={c['distance']:>2} "
        f"Items={c['items']}"
    )

# ==================================================
# SCHRITT 3: GENERALIZED SET COVERING MODELL
# ==================================================

model = pulp.LpProblem("RMFS_GSC", pulp.LpMinimize)

# binär variable
phi = pulp.LpVariable.dicts("phi", range(len(columns)), cat="Binary")
b = pulp.LpVariable("b", lowBound=0)  # max. Workload über alle Standorte

# Zielfunktion (8): M*b + sum(sigma_k * phi_k)
model += M * b + pulp.lpSum(columns[k]["distance"] * phi[k] for k in range(len(columns)))

# (9): jedes Item mind. 1x abgedeckt
for i in I:
    model += pulp.lpSum(
        phi[k] for k in range(len(columns)) if i in columns[k]["items"]
    ) >= 1, f"cover_item_{i}"

# (10): max. Q Spalten insgesamt
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

print("Status:", pulp.LpStatus[model.status])
print("Maximaler Workload b =", pulp.value(b))
print("Zielfunktionswert G =", pulp.value(model.objective))
print()

for k in range(len(columns)):
    if pulp.value(phi[k]) > 0.5:
        c = columns[k]
        print(f"GEWÄHLT: ID={c['id']} Loc={c['location']} Pods={c['pods']} W={c['workload']} D={c['distance']}")

