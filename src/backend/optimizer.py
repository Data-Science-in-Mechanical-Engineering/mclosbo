import time
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import GPy
from ruamel.yaml import YAML

import src.mclosbo_controller_optimization as mclosbo
import src.safeopt as safeopt

logger = logging.getLogger(__name__)
yaml = YAML()

from dataclasses import dataclass, field

class Parameter:
    def __init__(self, name, param_range, value=None):
        self.name = name
        self.param_range = param_range
        self.value = value

    def get_normalized_value(self):
        return (self.value - self.param_range[0]) / (
                self.param_range[1] - self.param_range[0]
        )

    def set_value_from_normalized(self, normalized_value):
        self.value = (
                normalized_value * (self.param_range[1] - self.param_range[0])
                + self.param_range[0]
        )

    def __str__(self):
        return f'{self.name}: {self.value}'

    def __repr__(self):
        return self.__str__()


@dataclass(unsafe_hash=True)
class DataPoint:
    parameters: list[Parameter]
    objective: float
    constraints: dict = field(default_factory=dict)

    def __repr__(self):
        return f'Params: {self.parameters} - Objective: {self.objective} - Constraints: {self.constraints}'


class Optimizer:
    def __init__(self, parameters, config, unscaled_list_of_gp_points, asynchronus=True):
        self.config = config
        self.parameters = parameters
        self.parameter_set = self.config["safebo_parameters"]["parameter_set"]
        self.opt = self.initialize_safeopt(unscaled_list_of_gp_points)
        self.asynchronus = asynchronus

    def normalize_point(self, parameters, point):
        return [
            (point[i] - parameters[i].param_range[0])
            / (parameters[i].param_range[1] - parameters[i].param_range[0])
            for i in range(len(parameters))
        ]

    def denormalize_point(self, parameters, point):
        return [
            point[i] * (parameters[i].param_range[1] - parameters[i].param_range[0])
            + parameters[i].param_range[0]
            for i in range(len(parameters))
        ]

    def get_next_parameters(self):
        """
        optimize next parameters
        """
        # assume the current data point is the last parameter set
        x_old = [
            self.parameters[i].get_normalized_value()
            for i in range(len(self.parameters))
        ]
        start = time.time()
        logger.info('Optimization started')
        if self.config["safebo_parameters"]["algorithm"] == "safeopt":
            next_point = self.opt.optimize()
        else:
            if self.asynchronus:
                next_point, index_next = self.opt.optimize(asynchronous=True, opt=x_old)
            else:
                next_point, index_next = self.opt.optimize(asynchronous=False)
            print(next_point)
        logger.info(f"Optimization completed. Elapsed time: {time.time() - start:.5f} seconds")
        for i in range(len(self.parameters)):
            self.parameters[i].set_value_from_normalized(next_point[i])
        return self.parameters

    def set_data_point(self, x, y):
        """
        set a new data point to the optimizer

        Parameters:
        x (list): list of parameters in Parameter class
        y (list): values of the objective function and the constraints
        """
        print(y)
        # determine the index of the new data point
        x = [parameter.get_normalized_value() for parameter in x]
        index_x = []
        index_x = np.argmin(np.linalg.norm(self.parameter_set - x, axis=1))
        if self.config["safebo_parameters"]["algorithm"] == "safeopt":
            self.opt.add_new_data_point(x, y)
        else:
            self.opt.add_new_data_point(x, index_x, y)

        if self.config["objective"]["GP"]["lengthscales"] != "fixed":
            self.opt.gps[0].optimize(messages=False, max_iters=10)
            logger.debug("Optimized objective GP")
            logger.debug(self.opt.gps[0].kern.lengthscale)
            logger.debug(self.opt.gps[0].kern.variance)
            #np.save(f"{self.config['base_config']['result_folder']}/gp_objective_{len(self.opt.gps[0].X)}.npy", self.opt.gps[0].param_array)
            #np.save(f"{self.config['base_config']['result_folder']}/gp_objective_{len(self.opt.gps[0].X)}_X.npy", self.opt.gps[0].X)
            #np.save(f"{self.config['base_config']['result_folder']}/gp_objective_{len(self.opt.gps[0].Y)}_Y.npy", self.opt.gps[0].Y)
            #np.save(f"{self.config['base_config']['result_folder']}/gp_objective_{len(self.opt.gps[0].X)}_full_obj.npy", self.opt.gps[0].param_array)

        for index, constraint in enumerate(self.config["constraints"]):
            if self.config["constraints"][constraint]["GP"]["lengthscales"] != "fixed":
                self.opt.gps[index + 1].optimize(messages=False, max_iters=10)
                logger.debug(self.opt.gps[index + 1].kern.lengthscale)
                logger.debug(self.opt.gps[index + 1].kern.variance)
                #np.save(f"{self.config['base_config']['result_folder']}/gp_constraint_{index}_{len(self.opt.gps[index + 1].X)}.npy", self.opt.gps[index + 1].param_array)
                #np.save(f"{self.config['base_config']['result_folder']}/gp_constraint_{index}_{len(self.opt.gps[index + 1].X)}_X.npy", self.opt.gps[index + 1].X)
                #np.save(f"{self.config['base_config']['result_folder']}/gp_constraint_{index}_{len(self.opt.gps[index + 1].Y)}_Y.npy", self.opt.gps[index + 1].Y)
            #np.save(f"{self.config['base_config']['result_folder']}/gp_constraint_{index}_{len(self.opt.gps[index + 1].Y)}_full.npy", self.opt.gps[index + 1].param_array)

    def get_scaled_list_for_gp(self, unscaled_list):
        scaled_list = []
        for i in range(len(unscaled_list)):
            scaled_list.append(self.normalize_point(self.parameters, unscaled_list[i]))
        return scaled_list

    def get_optimum(self):
        """
        returns the current optimum
        """
        # get the current optimum
        x_opt_index = self.opt.get_index_maximal_mean()
        x = self.parameter_set[x_opt_index]
        # denormalize the optimum
        return self.denormalize_point(self.parameters, x)

    def plot_lipschitz_kegel(self,lipschitz_bound, x, y, noise_bound, x_min=0, x_max=1, ax=None, safety_threshold=None):
        """
        Plots the Lipschitz cone given the parameters.

        Parameters:
        - L: float, the Lipschitz constant
        - x: float, the x-coordinate of the center point
        - y: float, the y-coordinate of the center point
        - E: float, the error term
        - x_min: float, the minimum x-coordinate
        - x_max: float, the maximum x-coordinate
        - ax: matplotlib.axes.Axes, the axes object to plot on
        - safety_threshold: float, threshold to filter out values

        Returns:
        - Y_values: numpy.ndarray, the filtered values of Y for each X value after applying the safety threshold
        - ax: matplotlib.axes.Axes, the updated axes object
        """
        if ax is None:
            ax = plt.gca()

        X_values = np.arange(x_min, x_max, 0.01)
        Y1_values = -lipschitz_bound * X_values + ((y - noise_bound) + lipschitz_bound * x)
        Y2_values = lipschitz_bound * X_values + ((y - noise_bound) - lipschitz_bound * x)
        Y_values = np.minimum(Y1_values, Y2_values)

        if safety_threshold is not None:
            valid_indices = Y_values > safety_threshold
            X_values, Y_values, Y1_values, Y2_values = [arr[valid_indices] for arr in [X_values, Y_values, Y1_values, Y2_values]]

        #make sure that for all values over y the min of Y1 and Y2 is taken, but only for all values over y
        # Calculate the minimum of Y1 and Y2
        Y_values = np.minimum(Y1_values, Y2_values,Y_values)

        # Adjust Y1_values and Y2_values to not exceed y
        Y1_values = np.where(Y1_values > y, Y_values, Y1_values)
        Y2_values = np.where(Y2_values > y, Y_values, Y2_values)

        # ax.plot(X_values, Y1_values, color='darkorange')
        # ax.plot(X_values, Y2_values, color='darkorange')
        Y_values = np.minimum(Y1_values, Y2_values,Y_values)
        ax.plot(X_values, Y_values, color='darkorange')

        return Y_values, ax

    def calculate_safe_range_point(self,lipschitz_bound, x, y, noise_bound, safety_threshold):
        """
        Calculates the safe range boundary points for a given line equation.

        Parameters:
        L (float): Lipschitz constant.
        x (float): The x-coordinate of the point.
        y (float): The y-coordinate of the point.
        E (float): The error term.
        safety_threshold (float): The threshold value for safety.

        Returns:
        tuple: A tuple containing the x-coordinate of the first boundary point and the x-coordinate of the second boundary point.
        """
        y_intercept_1 = (y - noise_bound) - lipschitz_bound * x
        y_intercept_2 = (y - noise_bound) + lipschitz_bound * x
        x_1_boundary = (safety_threshold - y_intercept_1) / lipschitz_bound
        x_2_boundary = (safety_threshold - y_intercept_2) / -lipschitz_bound
        return x_1_boundary, x_2_boundary

    def get_lipschlitz_intervall(self,lipschitz_bound, list_points, noise_bound, x_min=0, x_max=1,safety_threshold=None):
        """
        Calculates the Lipschitz interval for a given function.

        Parameters:
        L (float): Lipschitz constant.
        list_points (list): List of points [(x1, y1), (x2, y2), ...].
        E (float): Error tolerance.
        x_min (float, optional): Minimum x-value for the interval. Defaults to 0.
        x_max (float, optional): Maximum x-value for the interval. Defaults to 1.
        safety_threshold (float, optional): Safety threshold for calculating safe range point. Defaults to None.

        Returns:
        tuple: A tuple containing the minimum and maximum x-values of the Lipschitz interval.
        """
        
        list_boundaries_1 = []
        list_boundaries_2 = []
        for point in list_points:
            x_1_boundary, x_2_boundary = self.calculate_safe_range_point(lipschitz_bound, point[0], point[1], noise_bound, safety_threshold)
            list_boundaries_1.append(x_1_boundary)
            list_boundaries_2.append(x_2_boundary)
        x_min = max([x_min, min(list_boundaries_1)])
        x_max = min([x_max, max(list_boundaries_2)])
        return x_min, x_max

    def get_lipschlitz_interval_plot(self,lipschitz_bound, list_points, noise_bound, x_min=0, x_max=1,ax=None,safety_threshold=None):
        """
        Plots the Lipschitz interval for a given function.

        Parameters:
        lipschitz_bound (float): Lipschitz constant.
        list_points (list): List of points [(x1, y1), (x2, y2), ...].
        noise_bound (float): Error tolerance.
        x_min (float, optional): Minimum x-value for the interval. Defaults to 0.
        x_max (float, optional): Maximum x-value for the interval. Defaults to 1.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. Defaults to None.
        safety_threshold (float, optional): Safety threshold for calculating safe range point. Defaults to None.

        Returns:
        matplotlib.axes.Axes: The updated axes object.
        """
        if ax is None:
            ax = plt.gca()

        list_points = [(x, y) for x, y in list_points if x_min <= x <= x_max and y > safety_threshold]
        x_min, x_max = self.get_lipschlitz_intervall(lipschitz_bound, list_points, noise_bound, x_min, x_max,safety_threshold)
        ax.axvspan(x_min, x_max, alpha=0.1, color='black', label='Safe Set')
        return ax
    
    
    def get_projection_plot(self,param_index = 0, fig=None, axs=None):
        """
        Generates a projection plot for multidimensional optimization.

        Parameters:
        - config: Configuration dictionary
        - gps: List of GaussianProcessRegressor objects
        - opt: Optimization object
        - param_index: Index of the parameter to be projected
        - float_lipschlitz_const: Lipschitz constant for floating constraints (optional)

        Returns:
        - fig: Figure object containing the projection plot
        """
        def get_2D_plot(gp, ax, ylabel, dict_constraint=None, safety_threshold=None):
            """
            Plots a 2D graph with mean, standard deviation, current point, and previous points.

            Parameters:
            - gp: GaussianProcessRegressor object
            - ax: Axes object to plot the graph on
            - ylabel: Label for the y-axis
            - dict_constraint: Dictionary containing noise_bound and lipschitz_bound (optional)
            - safety_threshold: Threshold value for safety (optional)

            Returns:
            - ax: Updated Axes object with the plotted graph
            """
            X = gp.X
            y = gp.Y
            df = pd.DataFrame(X, columns=self.config["safebo_parameters"]["opt_params"])
            df["objective"] = y

            if dict_constraint is not None:
                noise_bound = dict_constraint["noise_bound"]
                lipschitz_bound = dict_constraint["lipschitz_bound"]
                

            list_points = [ (row[self.config["safebo_parameters"]["opt_params"][param_index]],row["objective"]) for index, row in df.iterrows()]
            projection = self.config["safebo_parameters"]["opt_params"][param_index]
            bound = self.config["safebo_parameters"]["bounds"][param_index]
            row = df.iloc[-1]
            fixed_dims_values = [row[key] for key in self.config["safebo_parameters"]["opt_params"] if
                                    key != projection]
            varying_dim = np.linspace(0, 1, 100)  # Vary 'tn' from 0 to 1( set the range as per your requirement)
            X_new = np.zeros((100, self.config["safebo_parameters"]["input_dim"]))
            for i in range(len(fixed_dims_values)):
                X_new[:, i + 1] = fixed_dims_values[i]
            X_new[:, 0] = varying_dim
            mean, variance = gp.predict(X_new)
            std_dev = np.sqrt(variance)

            ax.plot(varying_dim, mean, 'b-', label='Mean')
            ax.fill_between(varying_dim, mean[:, 0] - std_dev[:, 0], mean[:, 0] + std_dev[:, 0], alpha=0.2, label='Variance')
            ax.scatter(row[projection], row["objective"], c='r', zorder=5, label='Current Point')

            # plot the other points in the dataset the color should be the eucledean distance to the current point

            # Function to calculate Euclidean distance
            def euclidean_distance(row, last_row):
                return np.sqrt(np.sum((row - last_row) ** 2))

            # Extract the last row
            last_row = df[self.config["safebo_parameters"]["opt_params"][1:]].iloc[-1].values

            # Calculate distance for each row 
            df['distance'] = df[self.config["safebo_parameters"]["opt_params"][1:]].apply(
                lambda row: euclidean_distance(row.values, last_row), axis=1)
            if (len(self.config["safebo_parameters"]["opt_params"]) - 1) > 0:
                alphas = (df["distance"].values / np.sqrt(len(self.config["safebo_parameters"]["opt_params"]) - 1))
            else:
                alphas = np.zeros(len(df["distance"].values))
            alphas = 1 - alphas
            rgba_colors = np.zeros((len(df['distance'].values), 4))
            rgba_colors[:, 2] = 1.0
            rgba_colors[:, 3] = alphas
            if safety_threshold is not None:
                ax.plot(varying_dim, np.ones(len(varying_dim)) * safety_threshold, 'r-', label="Safety Threshold")
                _,ax = self.plot_lipschitz_kegel(lipschitz_bound = lipschitz_bound,x = row[projection],y = row["objective"],noise_bound = noise_bound,x_min=min(bound),x_max=max(bound),ax=ax,safety_threshold=safety_threshold)
                ax = self.get_lipschlitz_interval_plot(lipschitz_bound = lipschitz_bound,list_points = list_points,noise_bound = noise_bound,x_min=min(bound),x_max=max(bound),ax=ax,safety_threshold=safety_threshold)
            ax.scatter(X[:, param_index], y, zorder=4, label='Previous Points', color=rgba_colors)
            ax.set_xlabel(self.config["safebo_parameters"]["opt_params"][param_index])
            ax.set_ylabel(ylabel)
            return ax

        flag_initial_fig = True
        if fig is None:
            flag_initial_fig = False
            fig, axs = plt.subplots(len(self.gps),1)
            print(fig)
        axs[0] = get_2D_plot(self.gps[0], axs[0], "Objective")

        index = 1
        for constraint_name in self.config["constraints"]:
            
            axs[index] = get_2D_plot(self.gps[index], axs[index], constraint_name,self.config["constraints"][constraint_name],
                                        self.opt.fmin[index])
            index += 1
        
        if flag_initial_fig:
            return fig, axs
        else:
            fig.tight_layout()
            return fig

    def get_projection_plot_all_gps(self):
        """
        Returns a list of projection plots for all GPS parameters.

        Returns:
            list: A list of projection plots.
        """
        dict_of_plots = {}
        for param_name in self.config["safebo_parameters"]["opt_params"]:
            int_index = self.config["safebo_parameters"]["opt_params"].index(param_name)
            if self.config["parameters"][param_name]["optimized"]:
                dict_of_plots[param_name] = self.get_projection_plot(param_index=int_index)
            else:
                logger.debug(f"Parameter {param_name} is not to be optimized")
                continue
        return dict_of_plots

    def get_performance_plot(self):
        plot = None
        X = self.gps[0].X
        y = self.gps[0].Y
        df = pd.DataFrame(X, columns=self.config["safebo_parameters"]["opt_params"])
        df["objective"] = y

        plot = plt.figure()
        plt.plot(df["objective"].values)
        plt.title(f'Objective Function')
        plt.xlabel("Iteration")
        plt.ylabel('Objective Function')
        return plot

    def get_unscaled_list_of_gp_points(self):
        """
        returns unscaled list of points and measurements
        """
        points = [
            self.denormalize_point(self.parameters, point) for point in self.opt.x
        ]
        measurements = self.opt.y
        unscaled_list_of_gp_points = {"points": points, "measurements": measurements}
        return unscaled_list_of_gp_points

    def initialize_safeopt(self, unscaled_list_of_gp_points, gps=None):
        """
        initialize safe optimizer
        """
        logger.debug("Initializing Optimizer")
        # read initial results from csv
        config = self.config
        scaled_list = self.get_scaled_list_for_gp(unscaled_list_of_gp_points["points"])
        X = np.array(scaled_list)
        Y_1 = np.array(unscaled_list_of_gp_points["measurements"])[:, 0].reshape(-1, 1)
        print("y1", Y_1)

        index_x0 = [
            np.argmin(np.linalg.norm(self.parameter_set - i, axis=1)) for i in X
        ]
        if gps is not None:
            self.gps = gps
            for i, constraint in enumerate(config["constraints"]):
                Y = np.array(unscaled_list_of_gp_points["measurements"])[:, i + 1].reshape(-1, 1)
        else:
            # initialize objective gp and constraint gps
            constraint_gps = []
            objective_gp = self.init_gp(
                config["objective"]["GP"], X, Y_1, config["safebo_parameters"]["input_dim"]
            )
            for i, constraint in enumerate(config["constraints"]):
                Y = np.array(unscaled_list_of_gp_points["measurements"])[:, i + 1].reshape(-1, 1)
                constraint_gps.append(
                    self.init_gp(
                        config["constraints"][constraint]["GP"],
                        X,
                        Y,
                        config["safebo_parameters"]["input_dim"],
                    )
                )
            # append objective gp to constraint gps in front of list
            self.gps = [objective_gp] + constraint_gps
        
        safety_thresholds = [-np.inf] + [
                config["constraints"][constraint]["safety_threshold"]
                for constraint in config["constraints"]
            ]
        L = [1] + [
            config["constraints"][constraint]["lipschitz_bound"]
            for constraint in config["constraints"]
        ]
        noise = [0.01] + [
            config["constraints"][constraint]["noise_bound"]
            for constraint in config["constraints"]
        ]

        noise_std = 0.01

        # set up parameters for optimizer
        bounds = [(0, 1) for i in range(config["safebo_parameters"]["input_dim"])]
        card_D = config["safebo_parameters"]["grid_points_per_axis"] ** len(bounds)
        parameter_set = mclosbo.linearly_spaced_combinations(
            bounds, num_samples=config["safebo_parameters"]["grid_points_per_axis"]
        )

        # dictionary as input for optimizer (Lukas Kreisköther implementation)
        # index_x0 = index_x0[0]
        beta_dict = {
            "style": 2,
            "B": 30,
            "R": noise,
            "delta": 0.01,
            "lambda": noise_std ** 2,
            "noise_variance": noise_std ** 2,
            "card_D": card_D,
            "safety": "pure-lipschitz",
            "index_x0": index_x0,
            "y0": Y,
        }
        if config["safebo_parameters"]["algorithm"] == "safeopt":
            opt = safeopt.SafeOpt(
                self.gps,
                parameter_set,
                fmin=safety_thresholds,
                beta=config["safebo_parameters"]["beta"],
            )
        else:
            opt = mclosbo.SafeOpt(
                self.gps, parameter_set, lipschitz=L, fmin=safety_thresholds, beta_dict=beta_dict
            )

        [
            self.parameters[i].set_value_from_normalized(opt.x[-1, i])
            for i in range(len(self.parameters))
        ]

        return opt

    def init_gp(
            self, gp_config: dict, X: list, Y: list, input_dim: int
    ) -> GPy.models.GPRegression:
        print(gp_config["kernel"])
        if gp_config["lengthscales"] == "fixed":
            print("fixed lengthscale")
            if gp_config["kernel"] == "RBF":
                kernel = GPy.kern.RBF(
                    input_dim=input_dim,
                    variance=gp_config["variance"],
                    lengthscale=gp_config["lengthscale"]
                )
            elif gp_config["kernel"] == "Matern52":
                kernel = GPy.kern.Matern52(
                    input_dim=input_dim,
                    variance=gp_config["variance"],
                    lengthscale=gp_config["lengthscale"],
                )
            elif gp_config["kernel"] == "Matern32":
                kernel = GPy.kern.Matern32(
                    input_dim=input_dim,
                    variance=gp_config["variance"],
                    lengthscale=gp_config["lengthscale"],
                )
            else:
                kernel = GPy.kern.RBF(
                    input_dim=input_dim,
                    variance=gp_config["variance"],
                    lengthscale=gp_config["lengthscale"],
                )
                # set up prior mean function
            prior_mean = gp_config["prior_mean"]

            def constant(num):
                return prior_mean

            mf = GPy.core.Mapping(input_dim, 1)  # (parameter_anzahl, output_dimension)
            mf.f = constant
            mf.update_gradients = lambda a, b: None
            # set up first GP
            gp = GPy.models.GPRegression(
                X,
                Y[:, 0, None],
                kernel,
                noise_var=gp_config["noise_variance"] ** 2,
                mean_function=mf,
            )
        else:
            if gp_config["kernel"] == "RBF":
                kernel = GPy.kern.RBF(
                    input_dim=input_dim,
                    ARD=True
                )
            elif gp_config["kernel"] == "Matern52":
                kernel = GPy.kern.Matern52(
                    input_dim=input_dim,
                    ARD=True,
                )
            elif gp_config["kernel"] == "Matern32":
                kernel = GPy.kern.Matern32(
                    input_dim=input_dim,
                    ARD=True
                )
            else:
                kernel = GPy.kern.RBF(
                    input_dim=input_dim,
                    ARD=True
                )
            len_prior = GPy.priors.Gamma(3, 10)
            variance_prior = GPy.priors.Gamma(3, 2)

            # set up prior mean function
            prior_mean = gp_config["prior_mean"]

            def constant(num):
                return prior_mean

            mf = GPy.core.Mapping(input_dim, 1)  # (parameter_anzahl, output_dimension)
            mf.f = constant
            mf.update_gradients = lambda a, b: None
            gp = GPy.models.GPRegression(
                X,
                Y[:, 0, None],
                kernel,
                noise_var=gp_config["noise_variance"] ** 2,
                mean_function=mf,
            )
            gp.kern.lengthscale.set_prior(len_prior, warning=False)
            gp.kern.variance.set_prior(variance_prior, warning=False)

        return gp
