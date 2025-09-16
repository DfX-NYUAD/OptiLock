"""
A Dual Annealing global optimization algorithm
"""

import warnings

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy._lib._util import check_random_state
import math
# new added
from ec.impl.muxlink_fitness_function import MuxLinkFitnessFunction
import threading
import time
import copy

#__all__ = ['dual_annealing']

# define the kgss solution class - Zeng Wang
class KGSSInfo:
    """
    Class used to store information about the solution of the KGSS
    which could be sourced when we need the kgss_ori data structure,
    edge dict
    The main goal is to try our best to avoid wrong solution
    """
    def __init__(self, *args):
        self.kgss_ori = args[0]
        self.fitness_function = args[1]
        self.key_value = args[2]
        self.founded_pairs = copy.deepcopy(args[3]) # here we used it to update the found out pairs
        self.edge_dict = args[4]
        self.thread_num = args[5]
    def get_edge_dict(self):
        return self.edge_dict
    def get_founded_pairs(self):
        return self.founded_pairs
    def kgss_lock(self, sol_vector):
        kgss = self.kgss_ori
        key_value = self.key_value
        edge_dict = self.edge_dict

        kgss_data = []
        for i in range(0, len(sol_vector), 2):
            pair1 = int(math.floor(sol_vector[i]))
            pair2 = int(math.floor(sol_vector[i + 1]))
            # pair1 = int(math.ceil(sol_vector[i] / 2.) * 2)
            # pair2 = int(math.ceil(sol_vector[i + 1] / 2.) * 2)
            # print("pair1: ", pair1)
            # print("pair2: ", pair2)
            # find the corresponding edge
            pair_f1_g1 = edge_dict[pair1]
            pair_f2_g2 = edge_dict[pair2]
            key_temp = key_value[int(i / 2)]  # random.randint(0, 1)
            key_value += str(key_temp)

            kgss_temp = [str(pair_f1_g1[0]), str(pair_f2_g2[0]), str(pair_f1_g1[1]), str(pair_f2_g2[1]), str(key_temp),
                         int(i / 2)]

            kgss_data.append(kgss_temp)
        # get the length of current kgss data
        kgss_data_length = len(kgss_data)
        # append the foundpairs with the following index 
        founded_pairs = copy.deepcopy(self.founded_pairs) # copy with new memory
        for i in range(kgss_data_length, kgss_data_length +len(self.founded_pairs)):
            founded_pairs[i-kgss_data_length][5] = i # update the index
            founded_pairs[i-kgss_data_length][4] = self.key_value[i] # update the key value
            kgss_data.append(founded_pairs[i-kgss_data_length])
        kgss.data = kgss_data
        return kgss
    
    def kgss_delock(self, updated_kgss):
        sol_vec = []
        # print("what is the edge_dict")
        # print(self.edge_dict)
        # based on the founded pairs, remove the last elements
        updated_kgss_temp = updated_kgss
        founded_pairs_length = len(self.founded_pairs)
        # for i in range(len(updated_kgss)- founded_pairs_length, len(updated_kgss)):
        #     updated_kgss_temp.pop(updated_kgss_temp.index(updated_kgss[i]))
        updated_kgss = updated_kgss_temp[:len(updated_kgss)-founded_pairs_length]
        # updated_kgss = updated_kgss_temp

        for i in range(len(updated_kgss)):
            edge_pair1 = [int(updated_kgss[i][0]), int(updated_kgss[i][2])]
            edge_pair2 = [int(updated_kgss[i][1]), int(updated_kgss[i][3])]
            # found_num = 0
            # print("current iteration", i)
            edge_key1 = None
            edge_key2 = None

            for key, value in self.edge_dict.items():
                if value == edge_pair1:
                    edge_key1 = key
                    # found_num += 1
                    # found_num = 1
                    # print(edge_pair1)
                if value == edge_pair2:
                    edge_key2 = key
                    # found_num = 0
                    # found_num += 1
                    # print(edge_pair2)
            
            if edge_key1 is not None:
                sol_vec.append(edge_key1)
            if edge_key2 is not None:
                sol_vec.append(edge_key2)
            # if found_num == 2:
            #     print("correct")
            # else:
            #     print("false")
            # found_num = 0

                # else:
                #     print("diff", value)
        
        print(sol_vec)
        return np.array(sol_vec)

# define the class to implement the early stopping mechanism to stop the SA process
class EarlyStopping:
    def __init__(self, min_delta=0.005, patience=5, required_stops=1):
        self.min_delta = min_delta
        self.patience = patience
        self.required_stops = required_stops  # Number of times the stop condition must be met
        self.best_score = None
        self.worsened_score_times = 0
        self.stop_counter = 0  # Counter for how many times the stop condition has been met
        self.stop = False

    def __call__(self, current_value):
        if self.best_score is None:
            self.best_score = current_value
        elif current_value < self.best_score - self.min_delta:
            self.best_score = current_value
            self.worsened_score_times = 0
        else:
            self.worsened_score_times += 1

        # if the best score is reaching low than 0.5, we stop the process
        if self.best_score == 0:# < 0.5:
            self.stop = True

        if self.worsened_score_times >= self.patience:
            self.worsened_score_times = 0  # Reset the counter
            self.stop_counter += 1  # Increment the stop condition met counter
            if self.stop_counter >= self.required_stops:
                self.stop = True
        

class VisitingDistribution:
    """
    Class used to generate new coordinates based on the distorted
    Cauchy-Lorentz distribution. Depending on the steps within the strategy
    chain, the class implements the strategy for generating new location
    changes.
    Parameters
    ----------
    lb : array_like
        A 1-D NumPy ndarray containing lower bounds of the generated
        components. Neither NaN or inf are allowed.
    ub : array_like
        A 1-D NumPy ndarray containing upper bounds for the generated
        components. Neither NaN or inf are allowed.
    visiting_param : float
        Parameter for visiting distribution. Default value is 2.62.
        Higher values give the visiting distribution a heavier tail, this
        makes the algorithm jump to a more distant region.
        The value range is (1, 3]. It's value is fixed for the life of the
        object.
    rand_gen : {`~numpy.random.RandomState`, `~numpy.random.Generator`}
        A `~numpy.random.RandomState`, `~numpy.random.Generator` object
        for using the current state of the created random generator container.
    """
    TAIL_LIMIT = 1.e8
    MIN_VISIT_BOUND = 1.e-10

    def __init__(self, lb, ub, visiting_param, rand_gen):
        # if you wish to make _visiting_param adjustable during the life of
        # the object then _factor2, _factor3, _factor5, _d1, _factor6 will
        # have to be dynamically calculated in `visit_fn`. They're factored
        # out here so they don't need to be recalculated all the time.
        self._visiting_param = visiting_param
        self.rand_gen = rand_gen
        self.lower = lb
        self.upper = ub
        self.bound_range = ub - lb
        self.tried_visit = []

        # these are invariant numbers unless visiting_param changes
        self._factor2 = np.exp((4.0 - self._visiting_param) * np.log(
            self._visiting_param - 1.0))
        self._factor3 = np.exp((2.0 - self._visiting_param) * np.log(2.0)
                               / (self._visiting_param - 1.0))
        self._factor4_p = np.sqrt(np.pi) * self._factor2 / (self._factor3 * (
            3.0 - self._visiting_param))

        self._factor5 = 1.0 / (self._visiting_param - 1.0) - 0.5
        self._d1 = 2.0 - self._factor5
        self._factor6 = np.pi * (1.0 - self._factor5) / np.sin(
            np.pi * (1.0 - self._factor5)) / np.exp(gammaln(self._d1))

    # Zeng Wang- simplify the visiting function
    def visit_func_part(self, x, visits, upper_sample, lower_sample):
        visits[visits > self.TAIL_LIMIT] = self.TAIL_LIMIT * upper_sample
        visits[visits < -self.TAIL_LIMIT] = -self.TAIL_LIMIT * lower_sample
        x_visit = visits + x
        a = x_visit - self.lower
        b = np.fmod(a, self.bound_range) + self.bound_range
        x_visit = np.fmod(b, self.bound_range) + self.lower
        x_visit[np.fabs(
            x_visit - self.lower) < self.MIN_VISIT_BOUND] += 1.e-10
        return x_visit
    # Zeng Wang - simplify the visiting function1
    def visit_func_part1(self,step, dim, x, visit, x_visit):
        index = step - dim
        x_visit[index] = visit + x[index]
        a = x_visit[index] - self.lower[index]
        b = np.fmod(a, self.bound_range[index]) + self.bound_range[index]
        x_visit[index] = np.fmod(b, self.bound_range[
            index]) + self.lower[index]
        if np.fabs(x_visit[index] - self.lower[
            index]) < self.MIN_VISIT_BOUND:
            x_visit[index] += self.MIN_VISIT_BOUND
        return x_visit
    # Zeng Wang - change random seed for the random generator
    def rand_gen_seed(self):
        # change the rand_gen by setting the seed
        seed = self.rand_gen.randint(0, 100000)
        self.rand_gen = check_random_state(seed)
    def visiting(self, x, step, temperature, kgss_info):
        """ Based on the step in the strategy chain, new coordinated are
        generated by changing all components is the same time or only
        one of them, the new values are computed with visit_fn method
        """
        dim = x.size
        # founded_pairs = kgss_info.founded_pairs
        # founded_pairs_length = len(founded_pairs)*2
        if step < dim:
            # Changing all coordinates with a new visiting value
            visits = self.visit_fn(temperature, dim)
            upper_sample, lower_sample = self.rand_gen.uniform(size=2)
            visits[visits > self.TAIL_LIMIT] = self.TAIL_LIMIT * upper_sample
            visits[visits < -self.TAIL_LIMIT] = -self.TAIL_LIMIT * lower_sample
            # print("compare visits and visits", visits_temp == visits)
            x_visit = visits + x
            a = x_visit - self.lower
            b = np.fmod(a, self.bound_range) + self.bound_range
            x_visit = np.fmod(b, self.bound_range) + self.lower
            x_visit[np.fabs(
                x_visit - self.lower) < self.MIN_VISIT_BOUND] += 1.e-10
            # x_visit = self.visit_func_part(x, visits, upper_sample, lower_sample)
            # print(x_visit)
            # # here we also need to check if there is same gs
            # # unless we need to regenerate upper_sample and lower_sample
            kgss_lock = kgss_info.kgss_lock(x_visit)
            need_to_check = True
            false_check = 0
            while need_to_check: 
                if kgss_lock.check_same_gs_scope() or kgss_lock.check_cycles_scope():
                    kgss_edge_dict = kgss_info.get_edge_dict()
                    found_pairs = kgss_info.get_founded_pairs()
                    updated_kgss = kgss_lock.modify_same_gs_unchanged_scope(kgss_edge_dict, found_pairs)
                    if updated_kgss == None:
                        return None
                    # print("same element?")
                    # x_visit = kgss_info.kgss_delock(updated_kgss)
                    # print("check cycles?")
                    cycle_kgss = kgss_lock.check_cycle_pair_unchanged_scope(kgss_edge_dict, found_pairs)
                    if cycle_kgss == None:
                        return None
                    # updated_kgss = kgss_lock.modify_same_gs_selected(kgss_edge_dict)
                    x_visit = kgss_info.kgss_delock(cycle_kgss)
                if kgss_lock.check_f_in_g() or x_visit.tolist() in self.tried_visit: #or len(x_visit.tolist()) != 2*len(cycle_kgss):
                    # if f in g after we change, re-generate x_visit
                    visits = self.visit_fn(temperature, dim)
                    upper_sample, lower_sample = self.rand_gen.uniform(size=2)
                    x_visit = self.visit_func_part(x, visits, upper_sample, lower_sample)
                    kgss_lock = kgss_info.kgss_lock(x_visit)
                    continue
                else:
                    self.tried_visit.append(x_visit.tolist())
                    need_to_check = False
                if need_to_check == True:
                    false_check += 1
                    if false_check > 500:
                        return None
                
                
                







        else:
            # Changing only one coordinate at a time based on strategy
            # chain step
            x_visit = np.copy(x)
            visit = self.visit_fn(temperature, 1)[0]
            if visit > self.TAIL_LIMIT:
                visit = self.TAIL_LIMIT * self.rand_gen.uniform()
            elif visit < -self.TAIL_LIMIT:
                visit = -self.TAIL_LIMIT * self.rand_gen.uniform()
            index = step - dim
            x_visit[index] = visit + x[index]
            a = x_visit[index] - self.lower[index]
            b = np.fmod(a, self.bound_range[index]) + self.bound_range[index]
            x_visit[index] = np.fmod(b, self.bound_range[
                index]) + self.lower[index]
            if np.fabs(x_visit[index] - self.lower[
                    index]) < self.MIN_VISIT_BOUND:
                x_visit[index] += self.MIN_VISIT_BOUND
            # x_visit = self.visit_func_part1(step, dim, x, visit, x_visit)
            # here we also need to check if there is same gs
            kgss_lock = kgss_info.kgss_lock(x_visit)
            need_to_check1 = True
            false_check1 = 0
            while need_to_check1:
                if kgss_lock.check_same_gs_scope() or kgss_lock.check_cycles_scope():
                    kgss_edge_dict = kgss_info.get_edge_dict()
                    found_pairs = kgss_info.get_founded_pairs()
                    updated_kgss = kgss_lock.modify_same_gs_unchanged_scope(kgss_edge_dict, found_pairs)
                    if updated_kgss == None:
                        return None
                    # updated_kgss = kgss_lock.modify_same_gs_selected(kgss_edge_dict)
                    cycle_kgss = kgss_lock.check_cycle_pair_unchanged_scope(kgss_edge_dict, found_pairs)
                    if cycle_kgss == None:
                        return None
                    x_visit = kgss_info.kgss_delock(cycle_kgss)
                if kgss_lock.check_f_in_g() or x_visit.tolist() in self.tried_visit:
                    # if f in g after we change, re-generate x_visit
                    visit = self.visit_fn(temperature, 1)[0]
                    if visit > self.TAIL_LIMIT:
                        visit = self.TAIL_LIMIT * self.rand_gen.uniform()
                    elif visit < -self.TAIL_LIMIT:
                        visit = -self.TAIL_LIMIT * self.rand_gen.uniform()
                    x_visit = self.visit_func_part1(step, dim, x, visit, x_visit)
                    kgss_lock = kgss_info.kgss_lock(x_visit)
                    continue
                else:
                    self.tried_visit.append(x_visit.tolist())
                    need_to_check1 = False
                if need_to_check1 == True:
                    false_check1 += 1
                    if false_check1 > 500:
                        return None
        # print(x_visit)
        return x_visit

    def visit_fn(self, temperature, dim):
        """ Formula Visita from p. 405 of reference [2] """
        x, y = self.rand_gen.normal(size=(dim, 2)).T

        factor1 = np.exp(np.log(temperature) / (self._visiting_param - 1.0))
        factor4 = self._factor4_p * factor1

        # sigmax
        x *= np.exp(-(self._visiting_param - 1.0) * np.log(
            self._factor6 / factor4) / (3.0 - self._visiting_param))

        den = np.exp((self._visiting_param - 1.0) * np.log(np.fabs(y)) /
                     (3.0 - self._visiting_param))

        return x / den


class EnergyState:
    """
    Class used to record the energy state. At any time, it knows what is the
    currently used coordinates and the most recent best location.
    Parameters
    ----------
    lower : array_like
        A 1-D NumPy ndarray containing lower bounds for generating an initial
        random components in the `reset` method.
    upper : array_like
        A 1-D NumPy ndarray containing upper bounds for generating an initial
        random components in the `reset` method
        components. Neither NaN or inf are allowed.
    callback : callable, ``callback(x, f, context)``, optional
        A callback function which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and `context` has value in [0, 1, 2]
    """
    # Maximimum number of trials for generating a valid starting point
    MAX_REINIT_COUNT = 1000

    def __init__(self, lower, upper):
        self.ebest = None
        self.current_energy = None
        self.current_location = None
        self.xbest = None
        self.lower = lower
        self.upper = upper
        #self.callback = callback
    # use the generate sol to get the kgss data,
    # if there is same gs in the kgss, change the sol
    # generate no repeated gs x here
    def generate_sol(self, rand_gen, kgss_info):
        # Zeng Wang: generate the x_temp one by one to ignore the same gs
        used_gs = []
        x_all = []
        counter = 0
        lower_temp= self.lower[0:2]
        print("lower_temp in SA --Zeng", lower_temp)
        upper_temp = self.upper[0:2]
        print("upper_temp in SA --Zeng", upper_temp)
        while counter < len(self.lower)/2:
            x_temp = rand_gen.uniform(lower_temp, upper_temp, size=2)
            kgss_lock = kgss_info.kgss_lock(x_temp)
            gs_temp = kgss_lock.get_lock_fanout()
            if gs_temp not in used_gs:
                used_gs += gs_temp
                x_all += x_temp.tolist()
                counter += 1
        print("x_all in SA --Zeng", x_all)
        return np.array(x_all)

    def reset(self, func_wrapper, rand_gen, kgss_info, x0=None):
        """
        Initialize current location is the search domain. If `x0` is not
        provided, a random location within the bounds is generated.
        """
        # Zeng Wang:
        # if I set xo here, it will be assigned for the initial value
        # else, it will be assigned a random value in the bounds
        if x0 is None:
            print("reset debug in SA --Zeng")
            self.current_location = rand_gen.uniform(self.lower, self.upper,
                                                     size=len(self.lower))
            # self.current_location = self.generate_sol(rand_gen, kgss_info)
            """here is to check if the initial value is in the same gs"""
            kgss_lock = kgss_info.kgss_lock(self.current_location)
            # print("what is my kgss_lock", kgss_lock.data)
            # print("whatt is my founded pairs", kgss_info.founded_pairs)
            # print("kgss_lock length in SA --Zeng", len(kgss_lock.data))
            # check if there is same gs
            if kgss_lock.check_same_gs_scope() or kgss_lock.check_cycles_scope():
                print("same gs debug in SA --Zeng")
                # print("repeat gs in SA --Zeng", kgss_lock.check_all_same_gs())
                kgss_edge_dict = kgss_info.get_edge_dict()
                found_pairs = kgss_info.get_founded_pairs()
                updated_kgss = kgss_lock.modify_same_gs_unchanged_scope(kgss_edge_dict, found_pairs)
                if updated_kgss == None:
                    print("unsatisfied - modification in SA --Zeng")
                    return False
                cycle_kgss = kgss_lock.check_cycle_pair_unchanged_scope(kgss_edge_dict, found_pairs)
                if cycle_kgss == None:
                    print("unsatisfied - cycle in SA --Zeng")
                    return False
                # updated_kgss = kgss_lock.modify_same_gs_selected(kgss_edge_dict)
                self.current_location = kgss_info.kgss_delock(cycle_kgss)
                print("current_location length in SA --Zeng", len(self.current_location))
                # self.current_location = rand_gen.uniform(self.lower, self.upper,
                #                                      size=len(self.lower))
                # # self.current_location = self.generate_sol(rand_gen, kgss_info)
                # # print("new x0 in SA --Zeng", self.current_location)
                # kgss_lock = kgss_info.kgss_lock(self.current_location)
            # else:
            #     # also remove the added pairs into the 
        else:
            self.current_location = np.copy(x0)
        init_error = True
        reinit_counter = 0
        while init_error:
            self.current_energy = func_wrapper.fun(self.current_location)
            if self.current_energy is None:
                raise ValueError('Objective function is returning None')
            if (not np.isfinite(self.current_energy) or np.isnan(
                    self.current_energy)):
                if reinit_counter >= EnergyState.MAX_REINIT_COUNT:
                    init_error = False
                    message = (
                        'Stopping algorithm because function '
                        'create NaN or (+/-) infinity values even with '
                        'trying new random parameters'
                    )
                    raise ValueError(message)
                # here I do not change
                # since our fitness value is not infinity and Nan
                self.current_location = rand_gen.uniform(self.lower,
                                                         self.upper,
                                                         size=self.lower.size)
                reinit_counter += 1
                # print("reinit_counter in SA --Zeng", reinit_counter)
            else:
                init_error = False
            # If first time reset, initialize ebest and xbest
            if self.ebest is None and self.xbest is None:
                self.ebest = self.current_energy
                self.xbest = np.copy(self.current_location)
            # Otherwise, we keep them in case of reannealing reset
        # return None

    def update_best(self, e, x, context):
        self.ebest = e
        self.xbest = np.copy(x)
        # if self.callback is not None:
        #     val = self.callback(x, e, context)
        #     if val is not None:
        #         if val:
        #             return('Callback function requested to stop early by '
        #                    'returning True')

    def update_current(self, e, x):
        self.current_energy = e
        self.current_location = np.copy(x)

    def getCurrentState(self):
        return self.current_energy,self.current_location

    def getCurrentBestState(self):
        return self.ebest,self.xbest


class StrategyChain:
    """
    Class that implements within a Markov chain the strategy for location
    acceptance and local search decision making.
    Parameters
    ----------
    acceptance_param : float
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    visit_dist : VisitingDistribution
        Instance of `VisitingDistribution` class.
    func_wrapper : ObjectiveFunWrapper
        Instance of `ObjectiveFunWrapper` class.
    minimizer_wrapper: LocalSearchWrapper
        Instance of `LocalSearchWrapper` class.
    rand_gen : {None, int, `numpy.random.Generator`,
                `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
    energy_state: EnergyState
        Instance of `EnergyState` class.
    """
    lock = threading.Lock()
    def __init__(self, acceptance_param, visit_dist, func_wrapper,
                 minimizer_wrapper, rand_gen, energy_state,callback=None):
        # Local strategy chain minimum energy and location
        self.emin = energy_state.current_energy
        self.xmin = np.array(energy_state.current_location)
        # Global optimizer state
        self.energy_state = energy_state
        # Acceptance parameter
        self.acceptance_param = acceptance_param
        # Visiting distribution instance
        self.visit_dist = visit_dist
        # Wrapper to objective function
        self.func_wrapper = func_wrapper
        # Wrapper to the local minimizer
        self.minimizer_wrapper = minimizer_wrapper
        self.not_improved_idx = 0
        self.not_improved_max_idx = 1000
        self._rand_gen = rand_gen
        self.temperature_step = 0
        self.K = 100 * len(energy_state.current_location)
        self.callback=callback

    def accept_prob_val(self,e):
        pqv_temp = 1.0 - ((1.0 - self.acceptance_param) *
            (e - self.energy_state.current_energy) / self.temperature_step)
        if pqv_temp <= 0.:
            pqv = 0.
        else:
            pqv = np.exp(np.log(pqv_temp) / (
                1. - self.acceptance_param))
        return pqv

    def accept_reject(self, j, e, x_visit):
        accept = 0.0
        r = self._rand_gen.uniform()
        pqv_temp = 1.0 - ((1.0 - self.acceptance_param) *
            (e - self.energy_state.current_energy) / self.temperature_step)
        #print("DEBUG here 2:"+str(pqv_temp))
        pqv = pqv_temp
        if pqv_temp <= 0.:
            pqv = 0.
        # else:
        #     pqv = np.exp(np.log(pqv_temp) / (
        #         1. - self.acceptance_param))
        #print("DEBUG here 3:"+str(pqv))

        if r <= pqv:
            # We accept the new location and update state
            self.energy_state.update_current(e, x_visit)
            self.xmin = np.copy(self.energy_state.current_location)
            accept = 1.0

        # No improvement for a long time
        if self.not_improved_idx >= self.not_improved_max_idx:
            if j == 0 or self.energy_state.current_energy < self.emin:
                self.emin = self.energy_state.current_energy
                self.xmin = np.copy(self.energy_state.current_location)

        return pqv,accept
    
    def thread_run(self, j, temperature, kgss_info, thread_num, result_list):
        # generate x_visit
        x_visit = self.visit_dist.visiting(
                self.energy_state.current_location, j, temperature, kgss_info)
        
        # convert x_visit to kgss
        kgss_lock = kgss_info.kgss_lock(x_visit)
        with StrategyChain.lock:
            fitness_func = kgss_info.fitness_function
            # run the func on the generate x_visit
            e = fitness_func.evaluate_thread(thread_num, [kgss_lock])
            result_list[thread_num] = [e[0], x_visit]
        # print("result:", e, x_visit)

        # return e

    def run(self, step, temperature, kgss_info, early_stopping):
        self.temperature_step = temperature / float(step + 1)
        self.not_improved_idx += 1
        # change the range to (0.5*len(x),len(x))
        # for j in range(int(0.5 * len(self.energy_state.current_location)), len(self.energy_state.current_location)*2):
        for j in range(self.energy_state.current_location.size * 2):
            if j == 0:
                if step == 0:
                    self.energy_state_improved = True
                else:
                    self.energy_state_improved = False
            # Added by Zeng
            # multiple neighors check at the same time, and give us a better result
            thread_num = kgss_info.thread_num
            thread_list = []
            result_list = [None]*thread_num
            if thread_num > 0:
            # fitness_function = MuxLinkFitnessFunction(target_path, epochs=100)
                for thread_idx in range(thread_num):
                    thread = threading.Thread(target=self.thread_run, args=(j, temperature, kgss_info, thread_idx, result_list))
                    thread.start()
                    thread_list.append(thread)
                for thread in thread_list:
                    thread.join()
                e_list = []
                result_list_all = []
                print("result_list", result_list)
                for result in result_list:
                    e_list.append(float(result[0]))
                    result_list_all.append(result[1])
                max_value = min(e_list) # find the minimal one
                index_max = e_list.index(max_value)
                x_visit = result_list_all[index_max]
            elif thread_num == 0:

                x_visit = self.visit_dist.visiting(self.energy_state.current_location, j, temperature, kgss_info)
                if not isinstance(x_visit, np.ndarray):
                    return ('None value returned from visiting')
                print("Debug-SA-Zeng: x_visit:"+str(x_visit))
                # Calling the objective function
                e = self.func_wrapper.fun(x_visit)
                if e < self.energy_state.current_energy:
                    # We have got a better energy value
                    self.energy_state.update_current(e, x_visit)
                    accept_prob = 0.0 #self.accept_prob_val(e)
                    accept_flag= 1.0
                    if e < self.energy_state.ebest:
                        val = self.energy_state.update_best(e, x_visit, 0)
                        if val is not None:
                            if val:
                                return val
                        self.energy_state_improved = True
                        self.not_improved_idx = 0
                else:
                    # We have not improved but do we accept the new location?
                    print("DEBUG here")
                    print(str(e)+","+str(self.energy_state.current_energy))
                    accept_prob,accept_flag = self.accept_reject(j, e, x_visit)
                if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                    return ('Maximum number of function call reached '
                            'during annealing')
                
                if self.callback is not None:
                    cur_energy,cur_state = self.energy_state.getCurrentState()
                    best_energy,best_state = self.energy_state.getCurrentBestState()
                    data_dict = {
                        'cur_f' : cur_energy,
                        'cur_x' : cur_state,
                        'best_f' : best_energy,
                        'best_x' : best_state,
                        'tried_f': e,
                        'tried_x' : x_visit,
                        'temp' : temperature,
                        'a_prob' : accept_prob,
                        'a_flag' : accept_flag
                    }
                    val = self.callback(data_dict)
                if early_stopping is not None:
                    # update the early stopping with the current best score 
                    early_stopping(self.energy_state.ebest)
                    if early_stopping.stop:
                        # we need to return the early stopping message
                        return ('Early stopping condition met')

        # End of StrategyChain loop

    def local_search(self):
        # Decision making for performing a local search
        # based on strategy chain results
        # If energy has been improved or no improvement since too long,
        # performing a local search with the best strategy chain location
        if self.energy_state_improved:
            # Global energy has improved, let's see if LS improves further
            e, x = self.minimizer_wrapper.local_search(self.energy_state.xbest,
                                                       self.energy_state.ebest)
            if e < self.energy_state.ebest:
                self.not_improved_idx = 0
                val = self.energy_state.update_best(e, x, 1)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during local search')
        # Check probability of a need to perform a LS even if no improvement
        do_ls = False
        if self.K < 90 * len(self.energy_state.current_location):
            pls = np.exp(self.K * (
                self.energy_state.ebest - self.energy_state.current_energy) /
                self.temperature_step)
            if pls >= self._rand_gen.uniform():
                do_ls = True
        # Global energy not improved, let's see what LS gives
        # on the best strategy chain location
        if self.not_improved_idx >= self.not_improved_max_idx:
            do_ls = True
        if do_ls:
            e, x = self.minimizer_wrapper.local_search(self.xmin, self.emin)
            self.xmin = np.copy(x)
            self.emin = e
            self.not_improved_idx = 0
            self.not_improved_max_idx = self.energy_state.current_location.size
            if e < self.energy_state.ebest:
                val = self.energy_state.update_best(
                    self.emin, self.xmin, 2)
                if val is not None:
                    if val:
                        return val
                self.energy_state.update_current(e, x)
            if self.func_wrapper.nfev >= self.func_wrapper.maxfun:
                return ('Maximum number of function call reached '
                        'during dual annealing')


class ObjectiveFunWrapper:

    def __init__(self, func, maxfun=1e7, *args):
        self.func = func
        self.args = args
        # Number of objective function evaluations
        self.nfev = 0
        # Number of gradient function evaluation if used
        self.ngev = 0
        # Number of hessian of the objective function if used
        self.nhev = 0
        self.maxfun = maxfun

    def fun(self, x):
        self.nfev += 1
        return self.func(x, *self.args)


class LocalSearchWrapper:
    """
    Class used to wrap around the minimizer used for local search
    Default local minimizer is SciPy minimizer L-BFGS-B
    """

    LS_MAXITER_RATIO = 6
    LS_MAXITER_MIN = 100
    LS_MAXITER_MAX = 1000

    def __init__(self, search_bounds, func_wrapper, **kwargs):
        self.func_wrapper = func_wrapper
        self.kwargs = kwargs
        self.minimizer = minimize
        bounds_list = list(zip(*search_bounds))
        self.lower = np.array(bounds_list[0])
        self.upper = np.array(bounds_list[1])

        # If no minimizer specified, use SciPy minimize with 'L-BFGS-B' method
        if not self.kwargs:
            n = len(self.lower)
            ls_max_iter = min(max(n * self.LS_MAXITER_RATIO,
                                  self.LS_MAXITER_MIN),
                              self.LS_MAXITER_MAX)
            self.kwargs['method'] = 'L-BFGS-B'
            self.kwargs['options'] = {
                'maxiter': ls_max_iter,
            }
            self.kwargs['bounds'] = list(zip(self.lower, self.upper))

    def local_search(self, x, e):
        # Run local search from the given x location where energy value is e
        x_tmp = np.copy(x)
        mres = self.minimizer(self.func_wrapper.fun, x, **self.kwargs)
        if 'njev' in mres:
            self.func_wrapper.ngev += mres.njev
        if 'nhev' in mres:
            self.func_wrapper.nhev += mres.nhev
        # Check if is valid value
        is_finite = np.all(np.isfinite(mres.x)) and np.isfinite(mres.fun)
        in_bounds = np.all(mres.x >= self.lower) and np.all(
            mres.x <= self.upper)
        is_valid = is_finite and in_bounds

        # Use the new point only if it is valid and return a better results
        if is_valid and mres.fun < e:
            return mres.fun, mres.x
        else:
            return e, x_tmp

# here I want to add kgss, to make it work with kgss
# the main goal is to make it generate correct solution for each iteration
def dual_annealing(func, bounds, args=(), maxiter=1000,
                   minimizer_kwargs=None, initial_temp=5230.,
                   restart_temp_ratio=2.e-5, visit=2.62, accept=-5.0,
                   maxfun=1e7, seed=None, no_local_search=False,
                   callback=None, x0=None, local_search_options=None):
    """
    Find the global minimum of a function using Dual Annealing.
    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence, shape (n, 2)
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining bounds for the objective function parameter.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify the
        objective function.
    maxiter : int, optional
        The maximum number of global search iterations. Default value is 1000.
    minimizer_kwargs : dict, optional
        Extra keyword arguments to be passed to the local minimizer
        (`minimize`). Some important options could be:
        ``method`` for the minimizer method to use and ``args`` for
        objective function additional arguments.
    initial_temp : float, optional
        The initial temperature, use higher values to facilitates a wider
        search of the energy landscape, allowing dual_annealing to escape
        local minima that it is trapped in. Default value is 5230. Range is
        (0.01, 5.e4].
    restart_temp_ratio : float, optional
        During the annealing process, temperature is decreasing, when it
        reaches ``initial_temp * restart_temp_ratio``, the reannealing process
        is triggered. Default value of the ratio is 2e-5. Range is (0, 1).
    visit : float, optional
        Parameter for visiting distribution. Default value is 2.62. Higher
        values give the visiting distribution a heavier tail, this makes
        the algorithm jump to a more distant region. The value range is (1, 3].
    accept : float, optional
        Parameter for acceptance distribution. It is used to control the
        probability of acceptance. The lower the acceptance parameter, the
        smaller the probability of acceptance. Default value is -5.0 with
        a range (-1e4, -5].
    maxfun : int, optional
        Soft limit for the number of objective function calls. If the
        algorithm is in the middle of a local search, this number will be
        exceeded, the algorithm will stop just after the local search is
        done. Default value is 1e7.
    seed : {None, int, `numpy.random.Generator`,
            `numpy.random.RandomState`}, optional
        If `seed` is None (or `np.random`), the `numpy.random.RandomState`
        singleton is used.
        If `seed` is an int, a new ``RandomState`` instance is used,
        seeded with `seed`.
        If `seed` is already a ``Generator`` or ``RandomState`` instance then
        that instance is used.
        Specify `seed` for repeatable minimizations. The random numbers
        generated with this seed only affect the visiting distribution function
        and new coordinates generation.
    no_local_search : bool, optional
        If `no_local_search` is set to True, a traditional Generalized
        Simulated Annealing will be performed with no local search
        strategy applied.
    callback : callable, optional
        A callback function with signature ``callback(x, f, context)``,
        which will be called for all minima found.
        ``x`` and ``f`` are the coordinates and function value of the
        latest minimum found, and ``context`` has value in [0, 1, 2], with the
        following meaning:
            - 0: minimum detected in the annealing process.
            - 1: detection occurred in the local search process.
            - 2: detection done in the dual annealing process.
        If the callback implementation returns True, the algorithm will stop.
    x0 : ndarray, shape(n,), optional
        Coordinates of a single N-D starting point.
    local_search_options : dict, optional
        Backwards compatible flag for `minimizer_kwargs`, only one of these
        should be supplied.
    Returns
    -------
    res : OptimizeResult
        The optimization result represented as a `OptimizeResult` object.
        Important attributes are: ``x`` the solution array, ``fun`` the value
        of the function at the solution, and ``message`` which describes the
        cause of the termination.
        See `OptimizeResult` for a description of other attributes.
    Notes
    -----
    This function implements the Dual Annealing optimization. This stochastic
    approach derived from [3]_ combines the generalization of CSA (Classical
    Simulated Annealing) and FSA (Fast Simulated Annealing) [1]_ [2]_ coupled
    to a strategy for applying a local search on accepted locations [4]_.
    An alternative implementation of this same algorithm is described in [5]_
    and benchmarks are presented in [6]_. This approach introduces an advanced
    method to refine the solution found by the generalized annealing
    process. This algorithm uses a distorted Cauchy-Lorentz visiting
    distribution, with its shape controlled by the parameter :math:`q_{v}`
    .. math::
        g_{q_{v}}(\\Delta x(t)) \\propto \\frac{ \\
        \\left[T_{q_{v}}(t) \\right]^{-\\frac{D}{3-q_{v}}}}{ \\
        \\left[{1+(q_{v}-1)\\frac{(\\Delta x(t))^{2}} { \\
        \\left[T_{q_{v}}(t)\\right]^{\\frac{2}{3-q_{v}}}}}\\right]^{ \\
        \\frac{1}{q_{v}-1}+\\frac{D-1}{2}}}
    Where :math:`t` is the artificial time. This visiting distribution is used
    to generate a trial jump distance :math:`\\Delta x(t)` of variable
    :math:`x(t)` under artificial temperature :math:`T_{q_{v}}(t)`.
    From the starting point, after calling the visiting distribution
    function, the acceptance probability is computed as follows:
    .. math::
        p_{q_{a}} = \\min{\\{1,\\left[1-(1-q_{a}) \\beta \\Delta E \\right]^{ \\
        \\frac{1}{1-q_{a}}}\\}}
    Where :math:`q_{a}` is a acceptance parameter. For :math:`q_{a}<1`, zero
    acceptance probability is assigned to the cases where
    .. math::
        [1-(1-q_{a}) \\beta \\Delta E] < 0
    The artificial temperature :math:`T_{q_{v}}(t)` is decreased according to
    .. math::
        T_{q_{v}}(t) = T_{q_{v}}(1) \\frac{2^{q_{v}-1}-1}{\\left( \\
        1 + t\\right)^{q_{v}-1}-1}
    Where :math:`q_{v}` is the visiting parameter.
    .. versionadded:: 1.2.0
    References
    ----------
    .. [1] Tsallis C. Possible generalization of Boltzmann-Gibbs
        statistics. Journal of Statistical Physics, 52, 479-487 (1998).
    .. [2] Tsallis C, Stariolo DA. Generalized Simulated Annealing.
        Physica A, 233, 395-406 (1996).
    .. [3] Xiang Y, Sun DY, Fan W, Gong XG. Generalized Simulated
        Annealing Algorithm and Its Application to the Thomson Model.
        Physics Letters A, 233, 216-220 (1997).
    .. [4] Xiang Y, Gong XG. Efficiency of Generalized Simulated
        Annealing. Physical Review E, 62, 4473 (2000).
    .. [5] Xiang Y, Gubian S, Suomela B, Hoeng J. Generalized
        Simulated Annealing for Efficient Global Optimization: the GenSA
        Package for R. The R Journal, Volume 5/1 (2013).
    .. [6] Mullen, K. Continuous Global Optimization in R. Journal of
        Statistical Software, 60(6), 1 - 45, (2014).
        :doi:`10.18637/jss.v060.i06`
    Examples
    --------
    The following example is a 10-D problem, with many local minima.
    The function involved is called Rastrigin
    (https://en.wikipedia.org/wiki/Rastrigin_function)
    >>> from scipy.optimize import dual_annealing
    >>> func = lambda x: np.sum(x*x - 10*np.cos(2*np.pi*x)) + 10*np.size(x)
    >>> lw = [-5.12] * 10
    >>> up = [5.12] * 10
    >>> ret = dual_annealing(func, bounds=list(zip(lw, up)))
    >>> ret.x
    array([-4.26437714e-09, -3.91699361e-09, -1.86149218e-09, -3.97165720e-09,
           -6.29151648e-09, -6.53145322e-09, -3.93616815e-09, -6.55623025e-09,
           -6.05775280e-09, -5.00668935e-09]) # random
    >>> ret.fun
    0.000000
    """  # noqa: E501
    if x0 is not None and not len(x0) == len(bounds):
        raise ValueError('Bounds size does not match x0')

    lu = list(zip(*bounds))
    lower = np.array(lu[0])
    upper = np.array(lu[1])
    # Check that restart temperature ratio is correct
    if restart_temp_ratio <= 0. or restart_temp_ratio >= 1.:
        raise ValueError('Restart temperature ratio has to be in range (0, 1)')
    # Checking bounds are valid
    if (np.any(np.isinf(lower)) or np.any(np.isinf(upper)) or np.any(
            np.isnan(lower)) or np.any(np.isnan(upper))):
        raise ValueError('Some bounds values are inf values or nan values')
    # Checking that bounds are consistent
    if not np.all(lower < upper):
        raise ValueError('Bounds are not consistent min < max')
    # Checking that bounds are the same length
    if not len(lower) == len(upper):
        raise ValueError('Bounds do not have the same dimensions')

    # Wrapper for the objective function
    func_wrapper = ObjectiveFunWrapper(func, maxfun, *args)
    # Wrapper for the minimizer
    if local_search_options and minimizer_kwargs:
        raise ValueError("dual_annealing only allows either 'minimizer_kwargs' (preferred) or "
                         "'local_search_options' (deprecated); not both!")
    if local_search_options is not None:
        warnings.warn("dual_annealing argument 'local_search_options' is "
                      "deprecated in favor of 'minimizer_kwargs'",
                      category=DeprecationWarning, stacklevel=2)
        minimizer_kwargs = local_search_options

    # minimizer_kwargs has to be a dict, not None
    minimizer_kwargs = minimizer_kwargs or {}

    minimizer_wrapper = LocalSearchWrapper(
        bounds, func_wrapper, **minimizer_kwargs)
    # Zeng Wang - initialization of the kgss info
    kgss_info = KGSSInfo(*args)
    # Initialization of random Generator for reproducible runs if seed provided
    rand_state = check_random_state(seed)
    # OptimizeResult object to be returned
    optimize_res = OptimizeResult()
    # Initialization of the energy state
    #energy_state = EnergyState(lower, upper, callback)
    energy_state = EnergyState(lower, upper)
    # Zeng Wang: reset the energy state with kgss solution
    if energy_state.reset(func_wrapper, rand_state, kgss_info, x0) == False:
        print("reset failed in SA --Zeng")
        optimize_res.success = False
        optimize_res.x = None
        optimize_res.fun = None
        return optimize_res
    # Zeng Wang: add the function to check the early-stopping condition
    len_found_pairs = len(kgss_info.founded_pairs)
    len_key = len(list(kgss_info.key_value))
    # patience_value = (len_key - len_found_pairs)/2
    early_stopping = EarlyStopping(min_delta=0.005, patience=20, required_stops=1)
    # Minimum value of annealing temperature reached to perform
    # re-annealing
    temperature_restart = initial_temp * restart_temp_ratio
    # VisitingDistribution instance
    visit_dist = VisitingDistribution(lower, upper, visit, rand_state)
    # Strategy chain instance
    strategy_chain = StrategyChain(accept, visit_dist, func_wrapper,
                                   minimizer_wrapper, rand_state, energy_state,callback)
    need_to_stop = False
    iteration = 0
    message = []
    
    optimize_res.success = True
    optimize_res.status = 0

    t1 = np.exp((visit - 1) * np.log(2.0)) - 1.0
    # Run the search loop
    while(not need_to_stop):
        for i in range(maxiter):
            # Compute temperature for this step
            s = float(i) + 2.0
            t2 = np.exp((visit - 1) * np.log(s)) - 1.0
            temperature = initial_temp * t1 / t2
            if iteration >= maxiter:
                message.append("Maximum number of iteration reached")
                need_to_stop = True
                break
            # Need a re-annealing process?
            if temperature < temperature_restart:
                # Zeng Wang: reset the energy state with kgss solution
                energy_state.reset(func_wrapper, rand_state, kgss_info)
                break
            # starting strategy chain
            # Zeng Wang: add kgss_info as the last parameter
            val = strategy_chain.run(i, temperature, kgss_info, early_stopping)
            if val is not None:
                if "Early stopping" in val:
                    message.append(val)
                    need_to_stop = True
                    optimize_res.success = True
                    break
                else:
                    message.append(val)
                    need_to_stop = True
                    optimize_res.success = False
                    break
            # Possible local search at the end of the strategy chain
            if not no_local_search:
                val = strategy_chain.local_search()
                if val is not None:
                    message.append(val)
                    need_to_stop = True
                    optimize_res.success = False
                    break
            iteration += 1

    # Setting the OptimizeResult values
    optimize_res.x = energy_state.xbest
    optimize_res.fun = energy_state.ebest
    optimize_res.nit = iteration
    optimize_res.nfev = func_wrapper.nfev
    optimize_res.njev = func_wrapper.ngev
    optimize_res.nhev = func_wrapper.nhev
    optimize_res.message = message
    return optimize_res
