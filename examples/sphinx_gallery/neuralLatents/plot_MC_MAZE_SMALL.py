
import numpy as np
import plotly.graph_objects as go

from dandi.dandiapi import DandiAPIClient
from pynwb import NWBHDF5IO

import ssm.inference
import ssm.learning
import ssm.neural_latents.utils
import ssm.neural_latents.plotting


#%%
# Define auxiliary functions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

def add_events_vlines(fig, trials_df, events_names,
                      events_linetypes, events_colors):
    n_trials = trials_df.shape[0]
    for r in range(n_trials):
        for e, event_name in enumerate(events_names):
            fig.add_vline(x=trials_df.iloc[r][event_name],
                          line_dash=events_linetypes[e],
                          line_color=events_colors[e])


#%%
# Define variables
# ^^^^^^^^^^^^^^^^

# data
get_data_from_Dandi = True
dandiset_ID = "000140"
dandi_filepath = "sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
local_filepath = f"../../data/{dandiset_ID}/sub-Jenkins/sub-Jenkins_ses-small_desc-train_behavior+ecephys.nwb"
bin_size = 0.02

# plot
events_names = ["start_time", "target_on_time", "go_cue_time",
                "move_onset_time", "stop_time"]
events_linetypes = ["dot", "dash", "dashdot", "longdash", "solid"]
events_colors = ["white", "white", "white", "white", "white"]
cb_alpha = 0.3

# model
n_latents = 10

# estimation initial conditions
sigma_B = 0.1
sigma_Z = 0.1
sigma_Q = 0.1
sigma_R = 0.1
sigma_m0 = 0.1
sigma_V0 = 0.1

# estimation parameters
# max_iter = 5000
max_iter = 2
tol = 1e-1
vars_to_estimate = {"B": True, "Q": True, "Z": True, "R": True,
                    "m0": True, "V0": True, }

#%%
# Download data
# ^^^^^^^^^^^^^
if get_data_from_Dandi:
    with DandiAPIClient() as client:
        asset = client.get_dandiset(dandiset_ID,
                                    "draft").get_asset_by_path(dandi_filepath)
        s3_path = asset.get_content_url(follow_redirects=1, strip_query=True)
        io = NWBHDF5IO(s3_path, mode="r", driver="ros3")
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        trials_df = nwbfile.intervals["trials"].to_dataframe()
else:
    with NWBHDF5IO(local_filepath, 'r') as io:
        nwbfile = io.read()
        units_df = nwbfile.units.to_dataframe()
        trials_df = nwbfile.intervals["trials"].to_dataframe()


# n_clusters
n_clusters = units_df.shape[0]

#%%
# Bin spikes
# ^^^^^^^^^^

# continuous spikes times
continuous_spikes_times = [None for n in range(n_clusters)]
for n in range(n_clusters):
    continuous_spikes_times[n] = units_df.iloc[n]['spike_times']

binned_spikes, bin_edges = ssm.neural_latents.utils.bin_spike_times(
    spike_times=continuous_spikes_times, bin_size=bin_size)
bin_centers = (bin_edges[1:] + bin_edges[:-1])/2
transformed_binned_spikes = np.sqrt(binned_spikes + 0.5)

fig = go.Figure()
trace = go.Heatmap(x=bin_centers, z=transformed_binned_spikes,
                   colorbar=dict(title="<b>Sqrt(spike_count+0.5)</b>"))
fig.add_trace(trace)
add_events_vlines(fig=fig, trials_df=trials_df,
                  events_names=events_names,
                  events_linetypes=events_linetypes,
                  events_colors=events_colors)
fig.update_xaxes(title="Time (sec)")
fig.update_yaxes(title="Cluster Index")
fig

#%%
# Parameter learning using expectation maximization
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

B0 = np.diag(np.random.normal(loc=0, scale=sigma_B, size=n_latents))
Z0 = np.random.normal(loc=0, scale=sigma_Z, size=(n_clusters, n_latents))
Q0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_Q, size=n_latents)))
R0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_R, size=n_clusters)))
m0_0 = np.random.normal(loc=0, scale=sigma_m0, size=n_latents)
V0_0 = np.diag(np.abs(np.random.normal(loc=0, scale=sigma_V0, size=n_latents)))

optim_res = ssm.learning.em_SS_LDS(
    y=transformed_binned_spikes, B0=B0, Q0=Q0, Z0=Z0, R0=R0,
    m0_0=m0_0, V0_0=V0_0, max_iter=max_iter, tol=tol,
    vars_to_estimate=vars_to_estimate,
)

#%%
# Kalman filtering
# ^^^^^^^^^^^^^^^^

filter_res = ssm.inference.filterLDS_SS_withMissingValues_np(
    y=transformed_binned_spikes, B=optim_res["B"], Q=optim_res["Q"],
    m0=optim_res["m0"], V0=optim_res["V0"], Z=optim_res["Z"], R=optim_res["R"])

#%%
# Kalman smoothing
# ^^^^^^^^^^^^^^^^

smoothing_res = ssm.inference.smoothLDS_SS(
    B=optim_res["B"], xnn=filter_res["xnn"], Pnn=filter_res["Pnn"],
    xnn1=filter_res["xnn1"], Pnn1=filter_res["Pnn1"],
    m0=optim_res["m0"], V0=optim_res["V0"])

#%%
# Plot smoothed states
# ^^^^^^^^^^^^^^^^^^^^

o_means, o_covs = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
    means=smoothing_res["xnN"], covs=smoothing_res["PnN"], Z=optim_res["Z"])

fig = ssm.neural_latents.plotting.plot_latents(
    means=o_means,
    covs=o_covs,
    bin_centers=bin_centers,
    trials_df=trials_df,
    events_names=events_names,
    events_linetypes=events_linetypes,
    cb_alpha=cb_alpha,
    legend_pattern="smoothing_{:d}",
)

fig
