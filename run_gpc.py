import pickle

import os

from collections import OrderedDict

from tqdm import tqdm

from PIL import Image

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from torchvision import transforms as T
from torchvision.transforms import functional as Ft
from torchvision.models import get_model

import plotly.graph_objects as go
import plotly.io as pio

import pygpc


if __name__ == '__main__':
    print("Run gPC...")
    ## Load data ##
    results = np.load('./data/gPC/results.npy')
    coords = np.load('./data/gPC/coords.npy')
    with open('./data/gPC/parameters.pkl', 'rb') as file:
        parameters = loaded_ordered_dict = pickle.load(file)

    ## Run gPC ##
    grid = pygpc.RandomGrid(parameters_random=parameters, coords=coords)

    session_directory = './data/gpc/'
    results_filename = 'gpc'
    results_path = f'{session_directory}{results_filename}'
    session_extension = '.pkl'
    session_path = f'{results_path}{session_extension}'

    order = [6, 6, 6]
    order_max = sum(order)
    options = OrderedDict()
    options['method'] = 'reg'
    options['solver'] = 'Moore-Penrose'
    options['settings'] = None
    options['order'] = order
    options['order_max'] = order_max
    options['interaction_order'] = len(order)
    options['error_norm'] = 'relative'
    options['fn_results'] = results_path
    options['save_session_format'] = session_extension
    options['backend'] = 'omp'
    options['verbose'] = True

    algorithm = pygpc.Static_IO(parameters=parameters, options=options, grid=grid, results=results)

    session = pygpc.Session(algorithm=algorithm)

    session, coeffs, results = session.run()

    ## Postprocess gPC ##
    pygpc.get_sensitivities_hdf5(fn_gpc=results_path,
                                output_idx=None,
                                calc_sobol=True,
                                calc_global_sens=True,
                                calc_pdf=False,
                                algorithm="standard")

    sobol, gsens = pygpc.get_sens_summary(results_path, parameters)

    print(sobol)
    print(gsens)

    ## Plot approximation and samples ##
    print("Create plot...")
    # Set parameters
    plot_indices = [0, 1]
    fix_value = 10
    N = 100


    def scale(x, old_min, old_max, new_min, new_max):
        return new_min + (x - old_min) * (new_max - new_min) / (old_max - old_min)


    coords_parameters = list(session.parameters_random.items())
    fix_indices = list(set(range(len(coords_parameters))) - set(plot_indices))

    x_min = coords_parameters[plot_indices[0]][1].pdf_limits[0]
    x_max = coords_parameters[plot_indices[0]][1].pdf_limits[1]
    y_min = coords_parameters[plot_indices[1]][1].pdf_limits[0]
    y_max = coords_parameters[plot_indices[1]][1].pdf_limits[1]
    fix_min = coords_parameters[fix_indices[0]][1].pdf_limits[0]
    fix_max = coords_parameters[fix_indices[0]][1].pdf_limits[1]

    x_min_norm = coords_parameters[plot_indices[0]][1].pdf_limits_norm[0]
    x_max_norm = coords_parameters[plot_indices[0]][1].pdf_limits_norm[1]
    y_min_norm = coords_parameters[plot_indices[1]][1].pdf_limits_norm[0]
    y_max_norm = coords_parameters[plot_indices[1]][1].pdf_limits_norm[1]
    fix_min_norm = coords_parameters[fix_indices[0]][1].pdf_limits_norm[0]
    fix_max_norm = coords_parameters[fix_indices[0]][1].pdf_limits_norm[1]

    x_linspace = np.linspace(x_min, x_max, N)
    y_linspace = np.linspace(y_min, y_max, N)
    fix_values = np.full_like(x_linspace, fill_value=fix_value)

    x_norm_linspace = scale(x_linspace, x_min, x_max, x_min_norm, x_max_norm)
    y_norm_linspace = scale(y_linspace, y_min, y_max, y_min_norm, y_max_norm)
    fix_norm_values = scale(fix_values, fix_min, fix_max, fix_min_norm, fix_max_norm)

    xx_norm, yy_norm = np.meshgrid(x_norm_linspace, y_norm_linspace)

    coords_norm = np.stack(
        [
            xx_norm.reshape(-1),
            yy_norm.reshape(-1),
            np.full_like(xx_norm, fill_value=fix_norm_values[0]).reshape(-1),
        ],
        axis=1,
    )[:, np.argsort(plot_indices + fix_indices)]

    x_surface = x_linspace
    y_surface = y_linspace
    z_surface = (
        session.gpc[0]
        .get_approximation(coeffs=coeffs, x=coords_norm, output_idx=0)
        .reshape(N, N)
    )
    surface_plot = go.Surface(
        x=x_surface,
        y=y_surface,
        z=z_surface,
        colorscale="Plasma",
        showscale=True,
        name="gPC Approximation",
    )


    eps = (
        np.abs(np.max(coords[:, fix_indices[0]]) - np.min(coords[:, fix_indices[0]])) / 100
    )
    mask = (fix_value - eps < coords[:, fix_indices[0]]) & (
        coords[:, fix_indices[0]] < fix_value + eps
    )
    x_scatter = coords[mask, plot_indices[0]]
    y_scatter = coords[mask, plot_indices[1]]
    z_scatter = results[mask, 0]
    scatter_plot = go.Scatter3d(
        x=x_scatter,
        y=y_scatter,
        z=z_scatter,
        mode="markers",
        marker=dict(
            showscale=False,
            size=3,
            color="black",
            opacity=0.5,
        ),
        name="Ground Truth",
    )

    fig = go.Figure(data=[surface_plot, scatter_plot])

    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0, pad=0),
        scene_camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
        scene=dict(
            xaxis=dict(title=list(parameters.keys())[plot_indices[0]]),
            yaxis=dict(title=list(parameters.keys())[plot_indices[1]]),
            zaxis=dict(title="Probability"),
        ),
        legend=dict(
            title=dict(text="Legend"),
            x=0,
            y=0.1,
            xanchor="left",
            yanchor="middle",
            bgcolor="rgba(255, 255, 255, 0)",
        ),
    )
    fig.update_traces(showlegend=True)

    print("Write plot...")
    os.makedirs('./data/plots/', exist_ok=True)
    pio.write_image(fig=fig, file='./data/plots/welding.svg', format='svg', engine='orca')

    fig.show()
