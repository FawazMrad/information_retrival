# ============================================================
# scripts/make_architecture_diagram.py
#
# Draws the SOA architecture diagram (services + gateway + data) as a
# PNG you can drop straight into the report (requirement 7).
#
# Run:  python -m scripts.make_architecture_diagram
# ============================================================
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.5, edgecolor="#333", facecolor=color))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=9, weight="bold")


def arrow(ax, p1, p2, style="-|>", color="#555"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=14,
                                 linewidth=1.3, color=color))


def main():
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.set_xlim(0, 12); ax.set_ylim(0, 9); ax.axis("off")

    blue, green, orange, grey = "#CDE3F0", "#CFEAD6", "#FBE2C0", "#E8E8E8"

    box(ax, 4.5, 8.0, 3, 0.8, "User Interface\n(Streamlit, req 9)", blue)
    box(ax, 4.5, 6.4, 3, 0.8, "API Gateway\n(port 8000)", orange)

    box(ax, 0.4, 4.2, 2.5, 0.9, "Preprocessing\nService (8001)", green)
    box(ax, 3.3, 4.2, 2.5, 0.9, "Search / Retrieval\nService (8002)", green)
    box(ax, 6.2, 4.2, 2.5, 0.9, "Query Refinement\nService (8003)", green)
    box(ax, 9.1, 4.2, 2.5, 0.9, "Evaluation\nService (8004)", green)

    box(ax, 3.3, 2.2, 2.5, 0.9, "Inverted Index\nTF-IDF · BM25", grey)
    box(ax, 6.2, 2.2, 2.5, 0.9, "Word2Vec\nClusters", grey)
    box(ax, 9.1, 2.2, 2.5, 0.9, "Queries + qrels\n(ir_datasets)", grey)

    # UI <-> gateway
    arrow(ax, (6, 8.0), (6, 7.2), style="<|-|>")
    # gateway -> services
    for sx in (1.65, 4.55, 7.45, 10.35):
        arrow(ax, (6, 6.4), (sx, 5.1))
    # services -> data
    arrow(ax, (4.55, 4.2), (4.55, 3.1))
    arrow(ax, (7.45, 4.2), (7.45, 3.1))
    arrow(ax, (10.35, 4.2), (10.35, 3.1))
    # evaluation -> search (inter-service REST)
    arrow(ax, (9.1, 4.65), (5.8, 4.65), color="#C0392B")
    ax.text(7.4, 4.85, "REST", color="#C0392B", fontsize=8, ha="center")

    ax.text(6, 0.9, "Loose coupling · single responsibility per service · "
                    "REST communication · independently runnable",
            ha="center", fontsize=9, style="italic", color="#444")
    ax.set_title("IR System — Service-Oriented Architecture (SOA)", fontsize=14, weight="bold")

    out = os.path.join(config.DATA_DIR, "architecture_diagram.png")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print("Saved ->", out)


if __name__ == "__main__":
    main()
