import os

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json

from ruamel.yaml import YAML

from src.utils.visualization_utils import set_size, initialize_plot

params = initialize_plot('CDC')
mpl.rcParams.update(params)
yaml = YAML()
with open("src/utils/RWTHcolors.json") as json_file:
        c = json.load(json_file)

# Function to read CSV files and extract objective values
def extract_objective(folder_path):
    all_objectives = []
    all_constraints_1 = []
    all_constraints_2 = []
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.startswith("results_") and filename.endswith(".csv"):
                file_path = os.path.join(root, filename)
                df = pd.read_csv(file_path)
                all_objectives.append(df['objective'].values)
                all_constraints_1.append(df['constraint1'].values)
                all_constraints_2.append(df['constraint2'].values)
    return all_objectives, all_constraints_1, all_constraints_2

# Function to plot mean and standard deviation
def plot_experiment_lines(all_objectives, folder_name, ax, label, color = None):
    if int(folder_name[-1]) == 1:
        iterations = 16
    else:
        iterations = 51

    best_values_list = []
    
    for objectives in all_objectives:
        # determine best value so far for each round
        best_values = []
        best_std_values = []
        if len(objectives) == 1:
            best_values = [objectives[0] for i in range(iterations)]
        else:
            for i in range(len(objectives)):
                    best_values.append(np.nanmax(objectives[:i+1]))
        best_values_list.append(best_values)
    best_values_mean = np.mean(best_values_list, axis=0)
    best_std_values = np.std(best_values_list, axis=0)
    rounds = np.arange(0, len(best_values))
    
    ax.fill_between(rounds, best_values_mean - best_std_values, best_values_mean + best_std_values, alpha=0.2, color=color)
    ax.plot(rounds, best_values_mean, label=label, c=color)
    return ax


# Main function
def evaluate_simulation(path=None):
    if path is None:
        base_folder_path = os.path.join(os.getcwd(), 'results')
    else:
        base_folder_path = path        
    folders = [folder for folder in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, folder))]
    fig_x = 505.98/ 72.27
    fig_y = 505.98/4.5/ 72.27
    fig, ax = plt.subplots(1,3, figsize=(fig_x, fig_y), sharey=True)
    
    for folder_name in folders:
        folder_path = os.path.join(base_folder_path, folder_name)
        print(folder_path)
        detect_safety_violations(folder_path)
        all_objectives, all_constraints_1, all_constraints_2 = extract_objective(folder_path)
        
        if "safeopt" in folder_name:
            color = c["orange100"]
            label = "SafeOpt-MC"
        elif "t_1" in folder_name:
            color = c["gruen100"]
            label = "MCLosBO async"
        elif "t_2" in folder_name:
            color = c["tuerkis100"]
            label = "MCLosBO sync"
        elif "t_3" in folder_name:
            color = c["violett100"]
            label = "MCLosBO sync opt"
        elif "t_5" in folder_name:
            color =  c["blau100"]
            label = "MCLosBO async opt"    
        
        # determine safety violations
        safety_violations1 = 0
        safety_violations2 = 0
        for constraints_1 in all_constraints_1:
            safety_violations1 += np.sum(constraints_1 < -2)
        for constraints_2 in all_constraints_2:
            safety_violations2 += np.sum(constraints_2 < -0.2)
        print(folder_name)
        print(safety_violations1)
        print(safety_violations2)
        if int(folder_name[-1]) == 1:
            ax[0] = plot_experiment_lines(all_objectives, folder_name, ax[0], color = color, label= label)
        elif int(folder_name[-1]) == 2:
            ax[1] = plot_experiment_lines(all_objectives, folder_name, ax[1], color = color, label= label)
        elif int(folder_name[-1]) == 3:
            ax[2] = plot_experiment_lines(all_objectives, folder_name, ax[2], color = color, label= label)
    ax[0].set_ylabel('Objective')
    ax[1].set_xlim(0, 50)
    ax[2].set_xlim(0, 50)

    ax[0].set_xlim(0, 15)
    ax[0].set_xticks([0, 5, 10, 15])

    ax[0].set_ylabel('Objective')
    ax[0].set_ylabel('Objective')
    ax[0].set_xlabel('Iterations')
    ax[1].set_xlabel('Iterations')
    ax[2].set_xlabel('Iterations')
    ax[0].set_title('(a)')
    ax[1].set_title('(b)')
    ax[2].set_title('(c)')
    ax[0].legend()

    plt.show()
    
def detect_safety_violations(folder_path):
       for root, dirs, files in os.walk(folder_path):
        for filename in files:
                if filename.startswith("results_") and filename.endswith(".csv"):
                    file_path = os.path.join(root, filename)
                    df = pd.read_csv(file_path)
                    if np.sum(df['constraint1'] < -2) > 0:
                        print(file_path)
                    if np.sum(df['constraint2'] < -0.2) > 0:
                        print(file_path)
     
if __name__ == "__main__":
    evaluate_simulation()