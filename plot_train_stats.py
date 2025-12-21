import pandas as pd, matplotlib.pyplot as plt

df = pd.read_csv("stats/mh_quick.csv")
iters = df["iteration"]
plt.plot(iters, df["new_wins"]/(df["new_wins"]+df["prev_wins"]).fillna(0), label="new win rate")
plt.plot(iters, df["draws"], label="draws")
plt.step(iters, df["accepted"], label="accepted (0/1)")
plt.xlabel("Iteration"); plt.legend(); plt.tight_layout(); plt.show()

