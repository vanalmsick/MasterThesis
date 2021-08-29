import os, sys, warnings
import pandas as pd
import mlflow.keras
import mlflow
import shutil, time
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

# Working directory must be the higher .../app folder
if str(os.getcwd())[-3:] != 'app': raise Exception(f'Working dir must be .../app folder and not "{os.getcwd()}"')






def error_overview(y_pred, y_true):
    from scipy import stats
    data = np.array(y_pred).squeeze() - np.array(y_true).squeeze()
    out = [[('-inf', -2), stats.percentileofscore(data, -2)],
           [(-2,-1.5), stats.percentileofscore(data, -1.5) - stats.percentileofscore(data, -2)],
           [(-1.5,-1), stats.percentileofscore(data, -1) - stats.percentileofscore(data, -1.5)],
           [(-1,-0.5), stats.percentileofscore(data, -0.5) - stats.percentileofscore(data, -1)],
           [(-0.5,-0.25), stats.percentileofscore(data, -0.25) - stats.percentileofscore(data, -0.5)],
           [(-0.25,-0.15), stats.percentileofscore(data, -0.15) - stats.percentileofscore(data, -0.25)],
           [(-0.15,-0.10), stats.percentileofscore(data, -0.10) - stats.percentileofscore(data, -0.15)],
           [(-0.10,-0.05), stats.percentileofscore(data, -0.05) - stats.percentileofscore(data, -0.10)],
           [(-0.05,-0.025), stats.percentileofscore(data, -0.025) - stats.percentileofscore(data, -0.05)],
           [(-0.025,-0.01), stats.percentileofscore(data, -0.01) - stats.percentileofscore(data, -0.025)],
           [(-0.01,0), stats.percentileofscore(data, 0) - stats.percentileofscore(data, -0.01)],
           [(0,0.01), stats.percentileofscore(data, 0.01) - stats.percentileofscore(data, 0)],
           [(0.01,0.025), stats.percentileofscore(data, 0.025) - stats.percentileofscore(data, 0.01)],
           [(0.025,0.05), stats.percentileofscore(data, 0.05) - stats.percentileofscore(data, 0.025)],
           [(0.05,0.1), stats.percentileofscore(data, 0.1) - stats.percentileofscore(data, 0.05)],
           [(0.10,0.15), stats.percentileofscore(data, 0.15) - stats.percentileofscore(data, 0.10)],
           [(0.15,0.25), stats.percentileofscore(data, 0.25) - stats.percentileofscore(data, 0.15)],
           [(0.25,0.5), stats.percentileofscore(data, 0.5) - stats.percentileofscore(data, 0.25)],
           [(0.5,1), stats.percentileofscore(data, 1) - stats.percentileofscore(data, 0.5)],
           [(1,1.5), stats.percentileofscore(data, 1.5) - stats.percentileofscore(data, 1)],
           [(1.5,2), stats.percentileofscore(data, 2) - stats.percentileofscore(data, 1.5)],
           [(2, 'inf'), 1 - stats.percentileofscore(data, 2)]]
    out = np.array(out).T






def plot(examples_dict, normalization=True, y_pct=True):

    color_codes = ['#ff7f0e', '#58D68D', '#A569BD', '#40E0D0', '#922B21', '#CCCCFF', '#0E6655', '#1A5276']

    examples_len = examples_dict['examples_num']
    examples_comp = examples_dict['company'].tolist()

    y_hist = examples_dict['y_hist'].tolist()
    y_true = examples_dict['y_true'].tolist()

    t_idx = examples_dict['t_idx']
    time_steps = examples_dict['time_step']

    y_pred = []
    for i in range(examples_len):
        tmp_dict = {}
        for key, values in examples_dict['pred'].items():
            if len(examples_dict['pred'][key].shape) == 3:
                tmp_data = examples_dict['pred'][key][i, -1, :].tolist()
            elif len(examples_dict['pred'][key].shape) == 2:
                tmp_data = examples_dict['pred'][key][i, :].tolist()
            else:
                raise Exception(f'Strange number of {len(examples_dict["pred"][key].shape)} dimensions. Please check final model output of {key}!')
            tmp_dict[key] = tmp_data
        y_pred.append(tmp_dict)


    if len(examples_dict['pred']) > len(color_codes):
        raise Exception('Too few color codes defined. Please extend color_codes list!')


    fig = plt.figure(figsize=(12, 3 * examples_len))

    for i, time_step, comp, y_hist, y_true, t_idx, y_pred in zip(range(examples_len), time_steps, examples_comp, y_hist, y_true, t_idx, y_pred):
        x_t = [int(str(i)[:-2]) + (int(str(i)[-2:]) / 4) for i in t_idx]

        plt.subplot(examples_len, 1, i + 1)
        plt.ylabel(f'comp {comp} [{i}]')

        if normalization and 'norm_param' in examples_dict:
            scale = 1000000
            mean = examples_dict['norm_param'][i]['mean'][examples_dict['y_cols'][0]]
            std = examples_dict['norm_param'][i]['std'][examples_dict['y_cols'][0]]
        else:
            scale = 1
            mean = 0
            std = 1


        y_true_real = np.array(y_true) * std + mean
        y_hist_real = np.array(y_hist) * std + mean

        x_true = x_t[-len(y_true_real):]
        x_hist = x_t[:len(y_hist_real)]

        plt.plot(x_hist, (y_hist_real / scale), label='Historical', marker='.', zorder=-10)
        plt.plot([x_hist[-1]] + x_true, (np.array([y_hist_real.tolist()[-1]] + y_true_real.tolist()) / scale), label='True', marker='.', c='#A8A8A8', zorder=-11)

        j = 0
        for model, pred_y in y_pred.items():
            y_pred_real = np.array(pred_y) * std + mean
            plt.scatter(x_true[:len(y_pred_real)], (y_pred_real / scale), marker='X', edgecolors='k', label=f'{model} predictions', c=color_codes[j], s=64)
            j += 1

        if i == 0:
            plt.legend()

        #if i == examples_len - 1:
        #    plt.xticks(x_t, rotation='vertical')

        if y_pct:
            warnings.filterwarnings("ignore", message="FixedFormatter should only be used together with FixedLocator")
            vals = fig.get_axes()[i].get_yticks()
            fig.get_axes()[i].set_yticklabels(['{:,.2%}'.format(x) for x in vals])

        fig.get_axes()[i].xaxis.set_major_locator(MaxNLocator(integer=True))


    plt.show()




