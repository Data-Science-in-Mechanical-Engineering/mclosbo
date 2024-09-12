import os

import pandas as pd
import numpy as np

from ruamel.yaml import YAML

import src.mclosbo_controller_optimization as safeopt
from src.backend.run_experiment import evaluate_parameters
from src.backend.optimizer import Optimizer, Parameter, DataPoint

yaml = YAML()


#### UTILS ####
def update_configuration(config) -> dict:
    # check for how many parameters to optimize
    config["safebo_parameters"]["input_dim"] = 0
    config["safebo_parameters"]["opt_params"] = []
    config["safebo_parameters"]["fixed_params"] = []
    for parameter in config["parameters"]:
        if config["parameters"][parameter]["optimized"]:
            config["safebo_parameters"]["input_dim"] += 1
            config["safebo_parameters"]["opt_params"].append(parameter)
        else:
            config["safebo_parameters"]["fixed_params"].append(parameter)

    config["safebo_parameters"]["bounds"] = [
        (0, 1) for i in range(config["safebo_parameters"]["input_dim"])
    ]
    config["safebo_parameters"]["parameter_set"] = safeopt.linearly_spaced_combinations(
        config["safebo_parameters"]["bounds"],
        num_samples=config["safebo_parameters"]["grid_points_per_axis"],
    )
    return config

def load_parameters_from_config(config):
    """
    Get the parameters from the config file
    
    returns: optimized_parameters,
                fixed_parameters,
                baseline_parameters,
                initial_parameters
    """
    optimized_parameters = []
    fixed_parameters = []
    baseline_parameters = []
    initial_parameters = []

    for param in config["parameters"]:
        param_dict = config["parameters"][param]

        if param_dict["optimized"]:
            optimized_parameters.append(Parameter(param, param_dict["range"]))
        else:
            fixed_parameters.append(Parameter(param, param_dict["range"]))

        initial_parameters.append(Parameter(param, param_dict["range"], param_dict["initial_guess"]))
        baseline_parameters.append(Parameter(param, param_dict["range"], param_dict["baseline"]))

    return (optimized_parameters,
            fixed_parameters,
            baseline_parameters,
            initial_parameters)


def results_to_unscaled_gp_points(df, config, optimized_parameters):
    """
    Read the csv file with the previous measurements and parameters
    """
    params = [param.name for param in optimized_parameters]
    measurement_columns = ["objective"] + [constraint for constraint in config["constraints"]]
    df = df[df["log"].str.contains("baseline") == False]

    # Extract the required columns for points and measurements
    points = df[params].values.tolist()
    measurements = df[measurement_columns].values.tolist()

    # Format the data into the specified structure
    formatted_data = {
        "points": points,
        "measurements": measurements,
    }

    return formatted_data

def save_data_point(results_df, round, data_point: DataPoint, log="bo"):
    """
    Save the new data point to results database and save new csv results file

    returns: None
    """

    row = {"round": round - 1,
            **{param.name: param.value for param in data_point.parameters},
            "objective": data_point.objective,
            **data_point.constraints,
            "log": log}

    if results_df is None:
        results_df = pd.DataFrame([row])
    else:
        results_df = pd.concat([results_df, pd.DataFrame([row])], ignore_index=True)

    return results_df
    
def observe_data(parameters, config, seed=0):
        parameter_dict = parameter_to_dict(parameters, config)
        observation_dict = dict()
        for key in config["parameters"].keys():
            if not config["parameters"][key]["optimized"]:
                observation_dict[config["parameters"][key]["MABpath"]] = config["parameters"][key]["initial_guess"]
            else:
                observation_dict[config["parameters"][key]["MABpath"]] = parameter_dict[config["parameters"][key]["MABpath"]]
        #observation_dict["act_ext_pos_ctrl_tn"] = 1/parameter_dict["act_ext_pos_ctrl_Kn"]
        parameters = [Parameter(key, config["parameters"][key]["range"], observation_dict[config["parameters"][key]["MABpath"]]) for key in config["parameters"].keys()]
        results_dict=evaluate_parameters(observation_dict,seed=seed, config=config)
        data_point = DataPoint(parameters=parameters,
                               objective=-results_dict["objective"],
                               constraints={
                                   constraint: -results_dict[config["constraints"][constraint]["MABpath"]] for
                                   constraint in config["constraints"]})
        return data_point


def parameter_to_dict(parameters, config):
    """
    Convert a parameters list to a formatted dictionary that can be sent to the vehicle
    """
    formatted_data = {}
    for param in parameters:
        formatted_data[f"{config['parameters'][param.name]['MABpath']}"] = param.value
    return formatted_data

def get_plots(optimizer, counter, plots_file = "plot.png"):
    projection_plot = optimizer.get_projection_plot(fig = None, axs = None)
    print(projection_plot)
    projection_plot.savefig(plots_file.format('projection', counter))
    performance_plot = optimizer.get_performance_plot()
    performance_plot.savefig(plots_file.format('performance', counter))
    #get the plots for the other parameters
    dict_plots_all_gps = optimizer.get_projection_plot_all_gps()
    for key,value in dict_plots_all_gps.items():
        value.savefig(plots_file.format(f"projection_{key}", counter),bbox_inches='tight')



############################################ ASYNCHRONOUS EVALUATION ############################################

def asynchronous(config, results_path, seed=0):
    """
    async evaluation with mclosbo
    """
    np.random.seed(seed)


    (optimized_parameters,fixed_parameters,baseline_parameters,initial_parameters)=load_parameters_from_config(config)


    initial_datapoint1 = observe_data(initial_parameters, config, seed=seed)
    print(initial_datapoint1)
    results_df = save_data_point(None, 0, initial_datapoint1, log="init")
    results_df.to_csv(os.path.join(results_path, f"results_{seed}.csv"), index=False)
    unscaled_list_of_gp_points = results_to_unscaled_gp_points(results_df, config, optimized_parameters)

    opt = Optimizer(optimized_parameters, config, unscaled_list_of_gp_points)
    x_next= opt.get_next_parameters()

    initial_datapoint2 = observe_data(initial_parameters, config, seed=seed+100)
    results_df = save_data_point(results_df, 0, initial_datapoint2, log="init")
    unscaled_list_of_gp_points = results_to_unscaled_gp_points(results_df, config, optimized_parameters)

    for i in range(config["safebo_parameters"]["iterations"]-1):
        print(i)
        measurement = observe_data(x_next, config, seed)
        results_df = save_data_point(results_df, i, measurement, log="opt")
        unscaled_list_of_gp_points = results_to_unscaled_gp_points(results_df, config, optimized_parameters)
        constraints_values = [measurement.constraints[constraint] for constraint in config["constraints"]]
        print(constraints_values)
        
        #measurement.parameters = [param for param in measurement.parameters if param.name in [param.name for param in optimized_parameters]]
        opt.set_data_point([param for param in measurement.parameters if param.name in [param.name for param in optimized_parameters]], [[measurement.objective] + constraints_values])

        x_next = opt.get_next_parameters()
        print(x_next)

    plot_path = os.path.join(results_path, str(seed)+"plot_{}.png")
    get_plots(opt, 0, plot_path)
    return results_df


############################################ SYNCHRONOUS EVALUATION ############################################

def synchronous(config, results_path, seed=0):
    
    (optimized_parameters,fixed_parameters,baseline_parameters,initial_parameters)=load_parameters_from_config(config)

    initial_datapoint1 = observe_data(initial_parameters, config, seed=seed)
    results_df = save_data_point(None, 0, initial_datapoint1, log="init")
    results_df.to_csv(os.path.join(results_path, f"results_{seed}.csv"), index=False)
    unscaled_list_of_gp_points = results_to_unscaled_gp_points(results_df, config, optimized_parameters)
    opt = Optimizer(optimized_parameters, config, unscaled_list_of_gp_points, asynchronus=False)
    x_next = opt.get_next_parameters()

    for i in range(config["safebo_parameters"]["iterations"]):
        measurement = observe_data(x_next, config, seed)
        results_df = save_data_point(results_df, i, measurement, log="opt")
        unscaled_list_of_gp_points = results_to_unscaled_gp_points(results_df, config, optimized_parameters)
        constraints_values = [measurement.constraints[constraint] for constraint in config["constraints"]]
        
        #measurement.parameters = [param for param in measurement.parameters if param.name in [param.name for param in optimized_parameters]]
        opt.set_data_point([param for param in measurement.parameters if param.name in [param.name for param in optimized_parameters]], [[measurement.objective] + constraints_values])

        x_next = opt.get_next_parameters()
        print(x_next)
    
    get_plots(opt, 0, os.path.join(results_path,f"seed_{seed}_"+"plot_{}.png"))
    
    return results_df

def generate_safe_initial_parameters(config):
    """
    Generate initial parameters for the safe optimization
    """
    initial_parameters = []
    for param in config["parameters"]:
        if  config["parameters"][param]["optimized"]:
            param_dict = config["parameters"][param]
            random_value = np.random.uniform(param_dict["range"][0], param_dict["range"][1])
            initial_parameters.append(Parameter(param, param_dict["range"], random_value))
        else:
            param_dict = config["parameters"][param]
            initial_parameters.append(Parameter(param, param_dict["range"], param_dict["initial_guess"]))
    observation = observe_data(initial_parameters, config)
    # check if the observation violtes the constraints
    safety_violation = False
    for constraint in config["constraints"]:
        if observation.constraints[constraint] < config["constraints"][constraint]["safety_threshold"]:
            safety_violation = True
                
    if safety_violation:
        return generate_safe_initial_parameters(config)
    else:
        param_dict = parameter_to_dict(initial_parameters, config)
        for param in config["parameters"]:
            config["parameters"][param]["initial_guess"] = param_dict[config["parameters"][param]["MABpath"]]
        return config
    

experiments = dict()
experiments[10] = {"slurm_seed": 0, "experiment_name": "11", "algorithm": "mclosbo", "parameters": ["Kn"], "iterations": 15, "L1":10, "L2":3, "sync" : False, "hyperparameter_opt": False, "radom_safe_set": False} 
experiments[11] = {"slurm_seed": 0, "experiment_name": "12", "algorithm": "mclosbo", "parameters": ["Kn", "ssg"], "iterations": 15, "L1":10, "L2":3,"iterations": 50, "sync" : False, "hyperparameter_opt": False,  "radom_safe_set": False}
experiments[12] = {"slurm_seed": 0, "experiment_name": "13", "algorithm": "mclosbo", "parameters": ["Kn", "ssg", "tlat"], "iterations": 50, "L1":10, "L2":3, "sync" : False, "hyperparameter_opt": False,  "radom_safe_set": False} 

experiments[20] = {"slurm_seed": 0, "experiment_name": "21", "algorithm": "mclosbo", "parameters": ["Kn"], "iterations": 15,  "L1":10, "L2":3, "sync" : True, "hyperparameter_opt": False,  "radom_safe_set": False} 
experiments[21] = {"slurm_seed": 0, "experiment_name": "22", "algorithm": "mclosbo", "parameters": ["Kn", "ssg"], "iterations": 50,"L1":10, "L2":3, "sync" : True, "hyperparameter_opt": False,  "radom_safe_set": False}
experiments[22] = {"slurm_seed": 0, "experiment_name": "23", "algorithm": "mclosbo", "parameters": ["Kn", "ssg", "tlat"], "iterations": 50, "L1":10, "L2":3, "sync" : True, "hyperparameter_opt": False,  "radom_safe_set": False} 

experiments[30] = {"slurm_seed": 0, "experiment_name": "31", "algorithm": "mclosbo", "parameters": ["Kn"], "iterations": 15, "L1":10, "L2":3, "sync": True, "hyperparameter_opt": True, "radom_safe_set": False} 
experiments[31] = {"slurm_seed": 0, "experiment_name": "32", "algorithm": "mclosbo", "parameters": ["Kn", "ssg"], "iterations": 50, "L1":10, "L2":3, "sync" : True, "hyperparameter_opt": True,  "radom_safe_set": False}
experiments[32] = {"slurm_seed": 0, "experiment_name": "33", "algorithm": "mclosbo", "parameters": ["Kn", "ssg", "tlat"], "iterations": 50, "L1":10, "L2":3, "sync" : True, "hyperparameter_opt" : True, "radom_safe_set": False}

experiments[40] = {"slurm_seed": 0, "experiment_name": "41", "algorithm": "safeopt", "parameters": ["Kn"], "iterations": 15, "sync" : True, "L1":10, "L2":3, "hyperparameter_opt": False, "radom_safe_set": False} 
experiments[41] = {"slurm_seed": 0, "experiment_name": "42", "algorithm": "safeopt", "parameters": ["Kn", "ssg"], "iterations": 50, "sync" : True, "L1":10, "L2":3, "hyperparameter_opt": False, "radom_safe_set": False}
experiments[42] = {"slurm_seed": 0, "experiment_name": "43", "algorithm": "safeopt", "parameters": ["Kn", "ssg", "tlat"], "iterations": 50, "sync" : True, "L1":10, "L2":3, "hyperparameter_opt" : False, "radom_safe_set": False}

experiments[50] = {"slurm_seed": 0, "experiment_name": "51", "algorithm": "mclosbo", "parameters": ["Kn"], "iterations": 15, "sync" : False, "L1":10, "L2":3, "hyperparameter_opt": True, "radom_safe_set": False} 
experiments[51] = {"slurm_seed": 0, "experiment_name": "52", "algorithm": "mclosbo", "parameters": ["Kn", "ssg"], "iterations": 50, "sync" : False, "L1":10, "L2":3, "hyperparameter_opt": True, "radom_safe_set": False}
experiments[52] = {"slurm_seed": 0, "experiment_name": "53", "algorithm": "mclosbo", "parameters": ["Kn", "ssg", "tlat"], "iterations": 50, "sync" : False, "L1":10, "L2":3, "hyperparameter_opt" : True, "radom_safe_set": False}


def experiment(experiment_no, seed=100):
    
    experiment = experiments[experiment_no]
    # load config file
    np.random.seed(seed)
    config_file = "src/input/config.yaml"
    with open(config_file, 'r') as config_file:
        config = yaml.load(config_file)

    for parameter in config["parameters"]:
        if parameter in experiment["parameters"]:
            config["parameters"][parameter]["optimized"] = True
        else:
            config["parameters"][parameter]["optimized"] = False
    
    if experiment["radom_safe_set"]:
        config = generate_safe_initial_parameters(config)
            
    config["safebo_parameters"]["algorithm"] = experiment["algorithm"]
    config["safebo_parameters"]["iterations"] = experiment["iterations"]
    if experiment["hyperparameter_opt"]:
        config["constraints"]["constraint1"]["GP"]["lengthscales"]="flexible"
        config["constraints"]["constraint2"]["GP"]["lengthscales"]="flexible"
        config["objective"]["GP"]["lengthscales"]="flexible"
    else: 
        config["constraints"]["constraint1"]["GP"]["lengthscales"]="fixed"
        config["constraints"]["constraint2"]["GP"]["lengthscales"]="fixed"
        config["objective"]["GP"]["lengthscales"]="fixed"
    config["constraints"]["constraint1"]["lipschitz_bound"]=experiment["L1"]
    config["constraints"]["constraint2"]["lipschitz_bound"]=experiment["L2"]
    
    
    
    if os.path.isdir("results") == False:
        os.mkdir("results")
    results_path = os.path.join("results/", f"experiment_{experiment_no}_{experiment['algorithm']}_parameters_{len(experiment['parameters'])}")
    if os.path.isdir(results_path) == False:                                            # Create results folder if it does not exist
            os.mkdir(results_path)
    file = os.path.join(results_path, f"results_{seed}.csv")
    yaml.dump(config, open(results_path+"/config.yaml", "w"))
    config = update_configuration(config)
    print(experiment["sync"])
    if experiment["sync"]:
        results_df = synchronous(config, results_path, seed=seed)
    else:
        print("asynchronous")
        results_df = asynchronous(config, results_path, seed=seed)
    
    results_df.to_csv(file, index=False)
    

if __name__ == '__main__':
    for i in range(5):
        experiment(10, seed=i)
        experiment(11, seed=i)
        experiment(12, seed=i)
        experiment(20, seed=i)
        experiment(21, seed=i)
        experiment(22, seed=i)
        experiment(30, seed=i)
        experiment(31, seed=i)
        experiment(32, seed=i)
        experiment(40, seed=i)
        experiment(41, seed=i)
        experiment(42, seed=i)
        experiment(50, seed=i)
        experiment(51, seed=i)
        experiment(52, seed=i)
          
          
