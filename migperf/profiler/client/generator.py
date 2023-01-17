"""
Workload Generator (Simulator) for Benchmarking System.
Generating the requests (arrival time) according a fixed time duration and rate
by poisson distribution.
@author huangyz0918 (Yizheng Huang) and huaizhengzhang (Huaizheng Zhang)
Reference:
    - https://stackoverflow.com/questions/8592048/is-random-expovariate-equivalent-to-a-poisson-process
    - https://stackoverflow.com/questions/1155539/how-do-i-generate-a-poisson-process
    - https://github.com/marcoszh/MArk-Project/blob/master/experiments/request_sender.py
    - http://web.stanford.edu/class/archive/cs/cs109/cs109.1192/lectureNotes/8%20-%20Poisson.pdf
    -  http://web.mit.edu/modiano/www/6.263/lec5-6.pdf
"""
import random


class WorkloadGenerator:
    """
    Class of the requests generator.
    """

    @staticmethod
    def gen_arrival_time(duration=60 * 1, arrival_rate=5, seed=None):
        """
        Generating the arrival time according to the poisson distribution.
        :param duration: the requests sending duration (in second).
        :param arrival_rate: the average number of requests per second.
        :param seed: the random seed to reproduce the generated results.
        :return: a list of time to send requests.
        """
        start_time = 0
        arrive_time = []

        if seed is not None:
            random.seed(seed)

        while start_time < duration:
            start_time = start_time + random.expovariate(arrival_rate)
            arrive_time.append(start_time)

        return arrive_time