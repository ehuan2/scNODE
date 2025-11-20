import plotly.graph_objects as go
import colorsys


def generate_palette(n):
    """
    Generate n distinct HEX colors using HSV color space.
    """
    colors = []
    for i in range(n):
        h = i / n
        s = 0.65  # saturation
        v = 0.95  # brightness
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        colors.append("#%02x%02x%02x" % (int(r * 255), int(g * 255), int(b * 255)))
    return colors


def plot_sankey_from_labels(
    df, label_order=None, title="Cell Trajectory Sankey", path=None
):
    timepoints = list(df.columns)

    # Determine global label set
    if label_order is None:
        label_order = sorted(set(df.values.flatten()))

    # Assign colors to labels
    palette = generate_palette(len(label_order))
    label_to_color = {
        lab: palette[i % len(palette)] for i, lab in enumerate(label_order)
    }

    # Create nodes
    nodes = []
    node_index = {}
    idx = 0

    for t, tp in enumerate(timepoints):
        for label in label_order:
            node_name = f"{label}"
            node_index[(tp, label)] = idx
            nodes.append(node_name)
            idx += 1

    # Build flows
    sources = []
    targets = []
    values = []
    colors = []  # link colors

    for t in range(len(timepoints) - 1):
        t0, t1 = timepoints[t], timepoints[t + 1]
        counts = df.groupby([t0, t1]).size()

        for (label_from, label_to), count in counts.items():
            sources.append(node_index[(t0, label_from)])
            targets.append(node_index[(t1, label_to)])
            values.append(count)

            # Color link by source label
            colors.append(label_to_color[label_from])

    # Plot
    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(pad=20, thickness=20, label=nodes, color="rgba(0,0,0,0.75)"),
                link=dict(source=sources, target=targets, value=values, color=colors),
            )
        ]
    )

    fig.update_layout(title_text=title, font_size=14)
    if path is not None:
        fig.write_html(path)
