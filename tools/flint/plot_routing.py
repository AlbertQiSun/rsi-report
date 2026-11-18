"""Figure: Flint routing balance per layer and domain specialisation (data: research/flint_router_stats.json)."""
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
r = json.load(open(sys.argv[1])); out = sys.argv[2]
L = sorted(r["layers"], key=int); x = [int(l) for l in L]
doms = {"code": "#2563eb", "zh": "#dc2626", "en_prose": "#16a34a", "math": "#9333ea"}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for d, c in doms.items():
    ax[0].plot(x, [r["layers"][l][d]["norm_entropy"] for l in L], color=c, label=d, lw=1.6)
ax[0].set_ylim(0, 1); ax[0].axhline(1, color="grey", ls=":", lw=0.8)
ax[0].set_title("Expert-usage entropy / log(256)  (1 = perfectly balanced)"); ax[0].set_xlabel("layer"); ax[0].legend(fontsize=8)
for d, c in doms.items():
    ax[1].plot(x, [r["layers"][l][d]["top1_weight_mean"] for l in L], color=c, lw=1.6)
ax[1].axhline(1 / 6, color="grey", ls=":", lw=0.8); ax[1].text(x[0], 1 / 6 + 0.01, "uniform over 6 = 0.167", fontsize=8, color="grey")
ax[1].set_ylim(0, 1); ax[1].set_title("Mean top-1 routing weight (of 6 selected)"); ax[1].set_xlabel("layer")
pairs = list(r["layers"][L[0]]["jaccard_top20"]); base = (400 / 256) / (40 - 400 / 256)
vals = [np.mean([r["layers"][l]["jaccard_top20"][p] for l in L]) for p in pairs]
ax[2].barh(range(len(pairs)), vals, color="#64748b"); ax[2].set_yticks(range(len(pairs))); ax[2].set_yticklabels(pairs, fontsize=8)
ax[2].axvline(base, color="#dc2626", ls="--", lw=1); ax[2].text(base, len(pairs) - 0.4, f" random ≈ {base:.3f}", color="#dc2626", fontsize=8)
ax[2].set_xlim(0, max(vals) * 1.25); ax[2].set_title("Top-20 expert overlap between domains\n(Jaccard, mean over 47 MoE layers)")
fig.suptitle("Flint int4 — MoE routing dissection (exploratory: one ≤2048-token text per domain)", fontsize=10)
fig.tight_layout(); fig.savefig(out, dpi=150); print("saved", out, "random-Jaccard baseline", round(base, 4))
