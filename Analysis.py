import os
import re
import math
import itertools
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# Helpers / Normalization

CONTINENT_NORMALIZE = {
    "australia oceania": "Australia Oceania",
    "oceania": "Australia Oceania",
    "north america": "North America",
    "latin america": "Latin America",
    "europe": "Europe",
    "asia": "Asia",
    "africa": "Africa",
    "multiple": "Multiple",
}

GEOGRAPHY_NORMALIZE = {
    "australia oceania": "Australia Oceania",
    "oceania": "Australia Oceania",
    "north america": "North America",
    "latin america": "Latin America",
    "europe": "Europe",
    "asia": "Asia",
    "africa": "Africa",
    "global": "Global",
    "not mentioned": "Not mentioned",
}

def normalize_label(x: str, mapping: dict) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    key = s.lower().strip()
    return mapping.get(key, s)

def split_multi(x: str):
    """Split multi-valued fields like 'Europe; Asia' or 'Europe, Asia' into a list."""
    if pd.isna(x):
        return []
    s = str(x).strip()
    if not s:
        return []
    parts = re.split(r"[;,/|]+", s)
    return [p.strip() for p in parts if p.strip()]

def read_table():
    if os.path.exists("Table_cleaned.xlsx"):
        df = pd.read_excel("Table_cleaned.xlsx")
        src = "Table_cleaned.xlsx"
    elif os.path.exists("Table_cleaned.csv"):
        df = pd.read_csv("Table_cleaned.csv")
        src = "Table_cleaned.csv"
    else:
        raise FileNotFoundError("Could not find Table_cleaned.xlsx or Table_cleaned.csv in the current folder.")
    print(f"[OK] Loaded {src} with {len(df)} rows.")
    return df

def find_col(df, candidates):
    norm = {re.sub(r"\s+", " ", c.strip().lower()): c for c in df.columns}
    for cand in candidates:
        key = re.sub(r"\s+", " ", cand.strip().lower())
        if key in norm:
            return norm[key]
    raise KeyError(f"Could not find any of these columns: {candidates}")


# Load & filter data

df = read_table()

col_geo  = find_col(df, ["Study scale / case geography"])
col_aff  = find_col(df, ["Scholar affiliation continent(s)"])
col_conc = find_col(df, ["Conclusion"])

df = df[df[col_conc].astype(str).str.strip().str.lower().isin(["include", "maybe"])].copy()
print(f"[OK] After Include/Maybe filter: {len(df)} rows.")


# FIG 2 — Geographic distribution of study areas

def normalize_geography_value(x):
    """Normalize geography to a single category; multi-continent -> Global."""
    if pd.isna(x):
        return "Not mentioned"
    parts = split_multi(x)
    parts = [normalize_label(p, GEOGRAPHY_NORMALIZE) for p in parts if p]

    if len(parts) == 0:
        return "Not mentioned"
    if len(parts) > 1:
        return "Global"
    return parts[0]

geo_canon = df[col_geo].apply(normalize_geography_value)
geo_counts = Counter([g for g in geo_canon if g])

for k in ["Europe", "Asia", "North America", "Latin America", "Africa", "Australia Oceania", "Global", "Not mentioned"]:
    geo_counts.setdefault(k, 0)

print("[INFO] Geography counts:", dict(geo_counts))


def make_fig2(geo_counts: Counter, outpath="Fig2_geographic_distribution.png"):
    """
    Fig2: Geographic distribution of study areas (continent shading).
    Uses Cartopy Natural Earth admin_0 countries (no geodatasets dependency).
    """
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        from cartopy.io import shapereader
        import geopandas as gpd
    except Exception as e:
        print("\n[ERROR] This Fig2 method requires cartopy + geopandas.")
        print("Install with:")
        print("  pip3 install cartopy geopandas shapely pyproj fiona")
        raise e

    # Read Natural Earth countries
    shp_path = shapereader.natural_earth(
        resolution="110m",
        category="cultural",
        name="admin_0_countries"
    )

    world = gpd.read_file(shp_path)

    # Continent column detection
    cont_col = None
    for cand in ["CONTINENT", "continent", "REGION_UN", "Region_un"]:
        if cand in world.columns:
            cont_col = cand
            break
    if cont_col is None:
        raise KeyError(f"No usable continent column found. Columns: {list(world.columns)}")

    # Prefer CONTINENT if available
    if "CONTINENT" in world.columns:
        cont_col = "CONTINENT"

    def ne_to_our_cont(cont):
        # Natural Earth uses "South America" and "Oceania"
        if cont == "South America":
            return "Latin America"
        if cont == "Oceania":
            return "Australia Oceania"
        return cont

    world["our_continent"] = world[cont_col].apply(ne_to_our_cont)

    # Dissolve countries into continent shapes
    cont = world.dissolve(by="our_continent", as_index=False)

    # Attach counts (continents not present get 0)
    cont["count"] = cont["our_continent"].map(lambda c: geo_counts.get(c, 0)).fillna(0)

    # Plot with Cartopy projection
    fig = plt.figure(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax = plt.axes(projection=ccrs.Robinson())
    ax.set_global()
    ax.axis("off")
    ax.set_facecolor("white")

    # Custom light palette from white -> #EB4E46; absent -> white fill + black edge
    cont = cont.to_crs(epsg=4326)  # ensure lon/lat
    max_count = max(cont["count"].max(), 1)
    hi_rgb = mcolors.to_rgb("#EB4E46")

    def fill_for_count(c):
        if c <= 0:
            return "#ffffff"
        t = c / max_count  # 0..1
        # lighten overall: start at 0.25 (very light) up to 1.0 of target color
        t = 0.25 + 0.75 * t
        col = tuple((1 - t) * 1.0 + t * v for v in hi_rgb)  # mix with white
        return mcolors.to_hex(col)

    cont["fill_color"] = cont["count"].apply(fill_for_count)
    cont["edge_color"] = "#000000"

    # Draw polygons manually to respect per-row colors
    for _, row in cont.iterrows():
        geoms = row.geometry.geoms if hasattr(row.geometry, "geoms") else [row.geometry]
        ax.add_geometries(
            geoms,
            ccrs.PlateCarree(),
            facecolor=row["fill_color"],
            edgecolor=row["edge_color"],
            linewidth=0.4,
        )

    # Text block: present continents (desc), absent continents (middle), Global (second last), Not mentioned (last)
    present = [(k, geo_counts.get(k, 0)) for k in geo_counts if geo_counts.get(k, 0) > 0 and k not in ["Global", "Not mentioned"]]
    present = sorted(present, key=lambda x: x[1])  # ascending by count
    absent = sorted([k for k in geo_counts if geo_counts.get(k, 0) == 0 and k not in ["Global", "Not mentioned"]])
    # Order: Not mentioned (first), Global (second), absent continents (middle), present continents (ascending)
    order = ["Not mentioned", "Global"] + absent + [k for k, _ in present]

    fill_map = cont.set_index("our_continent")["fill_color"].to_dict()
    lines = []
    colors = []
    for k in order:
        lines.append(f"{k:<16} {geo_counts.get(k,0)}")
        colors.append(fill_map.get(k, "#000000"))

    # Render colored lines; fallback to black when white fill (absent)
    y0 = 0.10
    dy = 0.032
    for i, (line, color) in enumerate(zip(lines, colors)):
        ax.text(
            0.03,
            y0 + i * dy,
            line,
            transform=ax.transAxes,
            fontsize=14,
            va="bottom",
            ha="left",
            family="sans-serif",
            color=color if color != "#ffffff" else "#000000",
        )

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Fig2 -> {outpath}")

make_fig2(geo_counts)



# FIG 3 — Distribution of scholars & collaboration between continents


import networkx as nx

col_aff1 = find_col(df, ["Scholar affiliation continent(s)"])
col_aff2 = find_col(df, ["Continent2"])
col_aff3 = find_col(df, ["Continent3"])

def clean_cont(x):
    if pd.isna(x):
        return ""
    s = str(x).strip()
    if not s:
        return ""
    return normalize_label(s, CONTINENT_NORMALIZE)

def get_continent_set(row):
    conts = []
    c1 = clean_cont(row.get(col_aff1, ""))
    c2 = clean_cont(row.get(col_aff2, ""))
    c3 = clean_cont(row.get(col_aff3, ""))

    for c in [c1, c2, c3]:
        if c and c.lower() != "multiple":
            conts.append(c)

    # de-dupe
    conts = sorted(set([c for c in conts if c]))
    return conts

aff_sets = df.apply(get_continent_set, axis=1)

# DEBUG: show how many multi-continent rows exist
multi_rows = aff_sets.apply(lambda x: len(x) >= 2).sum()
print(f"[DEBUG] Rows with >=2 continents (should be >0 for edges): {multi_rows}")
print("[DEBUG] Example continent sets (first 10):")
print(aff_sets.head(10).tolist())

# Node counts: number of papers with at least one author from that continent
node_counts = Counter()
for conts in aff_sets:
    for c in conts:
        node_counts[c] += 1

# Edge weights: collaborations
edges_two = Counter()
edges_three_plus = Counter()

for conts in aff_sets:
    if len(conts) == 2:
        a, b = sorted(conts)
        edges_two[(a, b)] += 1
    elif len(conts) >= 3:
        # Any pair inside a 3+ continent paper is a "3+ collaboration tie"
        for a, b in itertools.combinations(sorted(conts), 2):
            edges_three_plus[(a, b)] += 1

print("[DEBUG] edges_two:", dict(edges_two))
print("[DEBUG] edges_three_plus:", dict(edges_three_plus))

def make_fig3(node_counts, edges_two, edges_three_plus, outpath="Fig3_cross_continent_collaboration.png"):
    import networkx as nx
    import matplotlib.patches as mpatches
    import textwrap

    nodes = sorted(node_counts.keys())
    G = nx.DiGraph()
    for n in nodes:
        G.add_node(n, weight=node_counts[n])

    # Fixed "paper-like" positions
    pos = {
        "Europe": (-0.15,  0.70),
        "North America": ( 0.95,  0.70),
        "Asia":          ( 0.95, -0.10),
        "Australia Oceania": (0.30, -0.70),
        "Latin America": (-0.75, -0.05),
        "Africa":        (-0.55, -0.75),
        "Not mentioned": (0.10, -0.05),
    }

    # Add missing nodes safely
    missing = [n for n in nodes if n not in pos]
    if missing:
        sub = nx.Graph()
        sub.add_nodes_from(missing)
        auto = nx.spring_layout(sub, seed=7)
        for k, v in auto.items():
            pos[k] = (v[0] * 0.4, v[1] * 0.4)

    # Colors (lightened fills + darker edges)
    base_color_map = {
        "Europe": "#f2c37b",
        "North America": "#b9c7ff",
        "Asia": "#7bdad6",
        "Australia Oceania": "#9fb8ff",
        "Latin America": "#f2a0a0",
        "Africa": "#f7d59a",
        "Not mentioned": "#cfcfcf",
    }

    def lighten(hex_color, factor=0.65):
        r, g, b = mcolors.to_rgb(hex_color)
        return ((1 - factor) + factor * r, (1 - factor) + factor * g, (1 - factor) + factor * b)

    def darker(hex_color, factor=0.75):
        r, g, b = mcolors.to_rgb(hex_color)
        return (r * factor, g * factor, b * factor)

    node_colors = [lighten(base_color_map.get(n, "#d9d9d9")) for n in nodes]
    node_edge_colors = [darker(base_color_map.get(n, "#d9d9d9")) for n in nodes]

    # Circle size proportional to node_counts (larger)
    max_w = max(node_counts.values()) if node_counts else 1

    def node_area(w):
        return 3000 + 32000 * (w / max_w)

    sizes = [node_area(node_counts[n]) for n in nodes]

    # Label wrapping + auto font sizing to fit inside circles
    def wrap_name(name: str) -> str:
        
        return "\n".join(textwrap.wrap(name, width=12)) if len(name) > 12 else name

    def label_fontsize(w):

        if w >= 20:
            return 16
        if w >= 10:
            return 14
        if w >= 5:
            return 13
        return 12

    max_edge = max([*edges_two.values(), *edges_three_plus.values()], default=1)

    def edge_width(w):
        return 0.5 + 1.5 * (w / max_edge)

    def edge_rad(a, b):
        # Straight line (no curvature)
        return 0.0

    fig, ax = plt.subplots(figsize=(11, 8))
    ax.set_axis_off()

    # Draw nodes
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=nodes,
        node_size=sizes,
        node_color=node_colors,
        edgecolors=node_edge_colors,
        linewidths=2.0,
        ax=ax,
    )

    # Draw edges as curved arrows (solid = 2-continent, dashdot = 3+)
    def draw_arrow(a, b, w, style="solid", alpha=0.75):
        arrow = mpatches.FancyArrowPatch(
            posA=pos[a], posB=pos[b],
            arrowstyle='-|>',              # arrow head
            mutation_scale=12,             # head size
            lw=edge_width(w),
            linestyle=style,
            color="black",
            alpha=alpha,
            connectionstyle="arc3,rad=0.0",
            shrinkA=18, shrinkB=18,        # keeps arrow from touching node edge
        )
        ax.add_patch(arrow)

        # edge label near midpoint, offset perpendicular to the line
        x1, y1 = pos[a]
        x2, y2 = pos[b]
        mx = (x1 + x2) / 2
        my = (y1 + y2) / 2
        dx, dy = x2 - x1, y2 - y1
        dlen = math.hypot(dx, dy)
        offx, offy = (0, 0)
        if dlen != 0:
            off_scale = 0.035
            offx = -dy / dlen * off_scale
            offy = dx / dlen * off_scale
        ax.text(mx + offx, my + offy, str(w), fontsize=10, ha="center", va="center", color="black")

    # Solid ties (two-continent collaborations), skip zero weights; draw once per pair
    for (a, b), w in edges_two.items():
        if w > 0:
            draw_arrow(a, b, w, style="solid", alpha=0.80)

    # Dashdot ties (3+ continent collaborations), skip zero weights; draw once per pair
    for (a, b), w in edges_three_plus.items():
        if w > 0:
            draw_arrow(a, b, w, style="dashdot", alpha=0.60)

    # Node labels
    for n in nodes:
        x, y = pos[n]
        fs = label_fontsize(node_counts[n])
        name_wrapped = wrap_name(n)
        ax.text(
            x, y,
            f"{name_wrapped}\n{node_counts[n]}",
            ha="center", va="center",
            fontsize=fs, weight="bold", color="#4a4a4a"
        )

    # Expand limits so nothing clips
    xs = [pos[n][0] for n in nodes]
    ys = [pos[n][1] for n in nodes]
    ax.set_xlim(min(xs) - 0.45, max(xs) + 0.45)
    ax.set_ylim(min(ys) - 0.45, max(ys) + 0.45)

    plt.tight_layout()
    plt.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] Saved Fig3 -> {outpath}")

make_fig3(node_counts, edges_two, edges_three_plus)



# Summary

n_total = len(df)
n_geo_reported = n_total - geo_counts.get("Not mentioned", 0)
n_multi_aff = sum(1 for s in aff_sets if len(s) >= 2)

print("\n====================")
print("SUMMARY (for report)")
print("====================")
print(f"Total included/maybe studies analysed: {n_total}")
print(f"Studies with reported case geography (not 'Not mentioned'): {n_geo_reported}")
print(f"Studies with cross-continent author teams (>=2 continents): {n_multi_aff}")
print("\nOutputs saved:")
print(" - Fig2_geographic_distribution.png")
print(" - Fig3_cross_continent_collaboration.png")
