import os
import pickle
import click
import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

from simulator import SoundSimulator, generate_random_map

DEFAULT_OPTIONS = {
    'nb_ex': 10000,
    'duration': 1000
}

HELP_MSG = {
    'nb_ex': 'Number of random examples to simulate',
    'duration': 'Number of iterations of the simulation'
}

@click.command()
@click.option("--nb-ex", "-n", type=int, 
              default=DEFAULT_OPTIONS["nb_ex"],
              help=HELP_MSG["nb_ex"])
@click.option("--duration", "-d", type=int, 
              default=DEFAULT_OPTIONS["duration"],
              help=HELP_MSG["duration"])
def launch(nb_ex, duration):

    map_size = (100, 100)

    def run_iteration(i):
        print(f"Simulating random example {i+1}/{nb_ex} ...")
        file_name = f"example_{i}.pickle"
        file_path = os.path.join("data", file_name)
        obstacle_map = generate_random_map(map_size, random_seed=0)
        simulation = SoundSimulator(map_size=(100, 100), 
                                    obstacle_map=obstacle_map, 
                                    duration= duration)
        simulation.run()
        spl = simulation.spl(integration_interval=10)
        with open(f"{file_path}", "wb") as f:
            pickle.dump((obstacle_map, spl), f)

    Parallel(n_jobs=num_cores)(delayed(run_iteration)(i) for i in range(nb_ex))

if __name__ == "__main__":
    launch()
    