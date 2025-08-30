
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import webcolors


def plot_latents(means, covs, bin_centers, trials_df, events_names,
                 events_linetypes, events_colors,
                 legend_pattern="filtering_{:d}", cb_alpha=0.3, colors=[],
                 xlabel="Time (sec)", ylabel="Latent Value"):
    if len(colors) == 0:
        colors = px.colors.qualitative.Plotly
    num_colors = len(colors)
    fig = go.Figure()
    n_states = means.shape[0]
    for i in range(n_states):
        color_rgb = webcolors.hex_to_rgb(colors[i % num_colors])
        color_pattern = \
            f"rgba({color_rgb[0]},{color_rgb[1]},{color_rgb[2]},{{:f}})"
        filter_means = means[i, 0, :]
        filter_stds = np.sqrt(covs[i, i, :])
        filter_ci_upper = filter_means + 1.96*filter_stds
        filter_ci_lower = filter_means - 1.96*filter_stds

        trace = go.Scatter(
            x=bin_centers, y=filter_means,
            mode="lines+markers",
            marker={"color": color_pattern.format(1.0)},
            name=legend_pattern.format(i),
            showlegend=True,
            legendgroup=legend_pattern.format(i),
        )
        trace_cb = go.Scatter(
            x=np.concatenate([bin_centers, bin_centers[::-1]]),
            y=np.concatenate([filter_ci_upper, filter_ci_lower[::-1]]),
            fill="toself",
            fillcolor=color_pattern.format(cb_alpha),
            line=dict(color=color_pattern.format(0.0)),
            showlegend=False,
            legendgroup=legend_pattern.format(i),
        )
        fig.add_trace(trace)
        fig.add_trace(trace_cb)

    add_events_vlines(fig=fig, trials_df=trials_df, events_names=events_names,
                      events_linetypes=events_linetypes,
                      events_colors=events_colors)

    fig.update_xaxes(title=xlabel)
    fig.update_yaxes(title=ylabel)
    return fig


def add_events_vlines(fig, trials_df, events_names,
                      events_linetypes, events_colors):
    n_trials = trials_df.shape[0]
    for r in range(n_trials):
        for e, event_name in enumerate(events_names):
            fig.add_vline(x=trials_df.iloc[r][event_name],
                          line_dash=events_linetypes[e],
                          line_color=events_colors[e])

