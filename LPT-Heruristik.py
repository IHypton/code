import pandas as pd
df = pd.read_csv("training_data.csv")

chosen = df[df["chosen"] == 1]
print("Anteil gewählter Spalten, die von LPT stammen:")
print(chosen["is_lpt_solution"].mean())

print("\nAnteil LPT-Spalten insgesamt, die gewählt wurden:")
lpt_cols = df[df["is_lpt_solution"] == 1]
print(lpt_cols["chosen"].mean())