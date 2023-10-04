import os
import pickle
import click
import multiprocessing
from joblib import Parallel, delayed

num_cores = multiprocessing.cpu_count()

from src.simulator import SoundSimulator
from src.utils import generate_random_map

DEFAULT_OPTIONS = {"nb_ex": 10, "duration": 200, "data_path": "."}

HELP_MSG = {
    "nb_ex": "Number of random examples to simulate",
    "duration": "Number of iterations of the simulation",
    "data_path": "Path to the folder in which simulation results are stored",
}


@click.command()
@click.option(
    "--nb-ex",
    "-n",
    type=int,
    default=DEFAULT_OPTIONS["nb_ex"],
    help=HELP_MSG["nb_ex"],
)
@click.option(
    "--duration",
    "-d",
    type=int,
    default=DEFAULT_OPTIONS["duration"],
    help=HELP_MSG["duration"],
)
@click.option(
    "--data-path",
    "-p",
    type=str,
    default=DEFAULT_OPTIONS["data_path"],
    help=HELP_MSG["data_path"],
)
def launch(nb_ex, duration, data_path):
    map_size = (100, 100)

    def run_iteration(iteration, random_seed):
        print(f"Simulating random example {iteration+1}/{nb_ex} ...")
        file_name = f"example_{iteration}.pickle"
        file_path = os.path.join(data_path, file_name)
        obstacle_map = generate_random_map(map_size, random_seed=random_seed)
        simulation = SoundSimulator(
            map_size=(100, 100), obstacle_map=obstacle_map, duration=duration
        )
        simulation.run()
        spl = simulation.spl(integration_interval=10)
        with open(f"{file_path}", "wb") as f:
            pickle.dump((obstacle_map, spl), f)

    Parallel(n_jobs=num_cores)(
        delayed(run_iteration)(i, i) for i in range(nb_ex)
    )


if __name__ == "__main__":
    launch()
