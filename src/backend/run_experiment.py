import pandas as pd
import numpy as np

# In the original paper, we used a simulator to evaluate the parameters.
# To still be able to run the code without the simulator, we evaluated the a fine grid of parameters on the simular and saved the results in a CSV file.
# We then interpolate the results from the CSV file to get the results for the given parameters.

def interpolate_points(parameters, csv_datapoints="experiment_data.csv"):
    """
    Interpolates data points from a CSV file to get a smooth surface for given parameters.

    Parameters:
    - parameters: Dictionary of parameters with their values for interpolation.
    - csv_datapoints: Path to the CSV file containing data points.

    Returns:
    - interpolated_value: Dictionary with interpolated values for targets not in parameters.
    """
    df = pd.read_csv(csv_datapoints)
    df = df.sort_values(by=list(parameters.keys()))
    
    # find the closest point in the data
    for key in parameters:
        df = df[df[key] >= parameters[key]-0.05]
        df = df[df[key] <= parameters[key]+0.05]
    interpolated_value = df.mean().to_dict()
    return interpolated_value

def add_noise(dict_results, dict_noise=None):
    """
    Adds random noise to the values in a dictionary.

    Parameters:
    - dict_results (dict): A dictionary containing the results.
    - dict_noise (dict, optional): A dictionary specifying the noise scale for each key. If not provided, the default noise scale is 0 for all keys.

    Returns:
    - dict: A dictionary with the noisy results.

    Example:
    >>> dict_results = {'A': 10, 'B': 20, 'C': 30}
    >>> dict_noise = {'A': 1, 'B': 2, 'C': 3}
    >>> add_noise(dict_results, dict_noise)
    {'A': 10.123, 'B': 19.987, 'C': 30.456}
    """
    if dict_noise is None:
        dict_noise = {key: 0 for key in dict_results.keys()}
    for key, value in dict_noise.items():
        dict_results[key] += np.random.normal(scale=value)
    return dict_results


def evaluate_parameters(parameters, seed=0, config=""):
    """
    Evaluates the parameters by performing interpolation and adding noise.

    Args:
        parameters (list): List of parameters to evaluate.
        seed (int, optional): Seed value for random number generation. Defaults to 0.
        config (str, optional): Configuration string. Defaults to "".

    Returns:
        dict: A dictionary containing the evaluated results.
    """
    print("Evaluating parameters:", parameters)
    for key in parameters: 
        parameters[key] = (parameters[key]-config["parameters"][key]["range"][0])/(config["parameters"][key]["range"][1]-config["parameters"][key]["range"][0])
    
    print("Normalized parameters:", parameters)
    dict_results = interpolate_points(parameters, csv_datapoints="src/input/experiment_data.csv")
    print(dict_results)
    dict_noise = {"objective":0.01, "constraint1":0.01, "constraint2":0.001}
    
    dict_results = add_noise(dict_results, dict_noise)
    return dict_results



