"""Knowledge graph from entities + co-occurrence."""
import os
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from config import PLOTS_DIR


# Articles & trailing junk that NER (especially spaCy) often glues onto
# entity spans: 'the University of Toronto', "Henry's", 'Henry later'.
_LEADING_ARTICLES = {"the", "a", "an", "el", "la", "le", "les", "los"}
_TRAILING_JUNK = {
    "later", "now", "then", "today", "yesterday", "tomorrow",
    "też", "także", "potem",
}


def _normalize_name(name):
    """Normalize an entity surface form for node merging."""
    if not name:
        return name
    s = name.strip()
    # Strip possessive 's / ’s
    s = re.sub(r"[\u2019']s\b", "", s)
    words = s.split()
    if not words:
        return name.strip()
    # Drop leading article
    if len(words) > 1 and words[0].lower() in _LEADING_ARTICLES:
        words = words[1:]
    # Drop trailing junk word
    while len(words) > 1 and words[-1].lower() in _TRAILING_JUNK:
        words = words[:-1]
    return " ".join(words).strip() or name.strip()


def build_graph(entities_per_sentence):
    """Build co-occurrence graph from a list of sentence-level entity lists.

    Args:
        entities_per_sentence: list of lists; each inner list contains entity dicts
            (with 'text' and 'label' keys) found in one sentence.
    """
    G = nx.Graph()
    for sent_ents in entities_per_sentence:
        # Normalize and dedupe within the sentence
        seen_in_sent = []
        for e in sent_ents:
            name = _normalize_name(e["text"])
            if not name:
                continue
            label = e.get("label", "MISC")
            qid = (e.get("link") or {}).get("qid") if e.get("link") else None
            if G.has_node(name):
                G.nodes[name]["count"] = G.nodes[name].get("count", 0) + 1
                if qid and not G.nodes[name].get("qid"):
                    G.nodes[name]["qid"] = qid
            else:
                G.add_node(name, label=label, qid=qid, count=1)
            if name not in seen_in_sent:
                seen_in_sent.append(name)
        # Edges between co-occurring entities
        for i in range(len(seen_in_sent)):
            for j in range(i + 1, len(seen_in_sent)):
                a, b = seen_in_sent[i], seen_in_sent[j]
                if a == b:
                    continue
                if G.has_edge(a, b):
                    G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
                else:
                    G.add_edge(a, b, weight=1)
    return G


_LABEL_COLORS = {
    "PERSON": "#ff7f0e",
    "PER": "#ff7f0e",
    "persName": "#ff7f0e",
    "ORG": "#1f77b4",
    "orgName": "#1f77b4",
    "GPE": "#2ca02c",
    "LOC": "#2ca02c",
    "placeName": "#2ca02c",
    "DATE": "#9467bd",
    "TIME": "#9467bd",
    "MONEY": "#bcbd22",
    "PRODUCT": "#17becf",
    "EVENT": "#e377c2",
    "MISC": "#7f7f7f",
}


def _layout(G):
    """Pick a layout that keeps disconnected components close together
    (spring_layout scatters them across the canvas)."""
    n = G.number_of_nodes()
    if n <= 1:
        return nx.spring_layout(G, seed=42)
    if nx.is_connected(G) and 2 <= n <= 25:
        try:
            return nx.kamada_kawai_layout(G)
        except Exception:
            pass
    # Lay out each connected component on its own grid cell, then merge.
    components = list(nx.connected_components(G))
    if len(components) == 1:
        return nx.spring_layout(G, seed=42, k=1.2 / max(1, n) ** 0.5)

    pos = {}
    cols = int(len(components) ** 0.5 + 0.999)
    for idx, comp in enumerate(components):
        sub = G.subgraph(comp)
        try:
            sub_pos = nx.kamada_kawai_layout(sub)
        except Exception:
            sub_pos = nx.spring_layout(sub, seed=42)
        cx, cy = idx % cols, idx // cols
        for node, (x, y) in sub_pos.items():
            pos[node] = (x * 0.6 + cx * 1.8, y * 0.6 + cy * 1.8)
    return pos


def plot_graph(G, title="Knowledge Graph", filename=None):
    if len(G.nodes) == 0:
        return None
    os.makedirs(PLOTS_DIR, exist_ok=True)
    if filename is None:
        filename = f"knowledge_graph_{int(time.time())}.png"
    out_path = os.path.join(PLOTS_DIR, filename)

    fig, ax = plt.subplots(figsize=(11, 8))
    pos = _layout(G)

    used_labels = sorted({
        G.nodes[n].get("label", "MISC") for n in G.nodes
    })
    colors = [
        _LABEL_COLORS.get(G.nodes[n].get("label", "MISC"), "#7f7f7f")
        for n in G.nodes
    ]
    sizes = [600 + 400 * G.nodes[n].get("count", 1) for n in G.nodes]

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=colors, node_size=sizes,
        alpha=0.9, edgecolors="white", linewidths=1.5,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=10, font_weight="bold",
    )
    weights = [G[u][v].get("weight", 1) for u, v in G.edges]
    nx.draw_networkx_edges(
        G, pos, ax=ax,
        width=[1.0 + 0.8 * w for w in weights],
        alpha=0.55, edge_color="#555555",
    )
    # Edge labels for weight > 1 (otherwise too noisy)
    edge_labels = {
        (u, v): str(G[u][v]["weight"])
        for u, v in G.edges if G[u][v].get("weight", 1) > 1
    }
    if edge_labels:
        nx.draw_networkx_edge_labels(
            G, pos, ax=ax, edge_labels=edge_labels, font_size=8,
        )

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D(
            [0], [0], marker="o", linestyle="",
            markerfacecolor=_LABEL_COLORS.get(lbl, "#7f7f7f"),
            markeredgecolor="white", markersize=10, label=lbl,
        )
        for lbl in used_labels
    ]
    if legend_handles:
        ax.legend(
            handles=legend_handles, loc="upper right",
            frameon=True, fontsize=9, title="Entity type",
        )

    ax.set_title(title, fontsize=13)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def graph_stats(G):
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "density": nx.density(G) if G.number_of_nodes() > 1 else 0.0,
        "components": nx.number_connected_components(G),
    }
