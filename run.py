import os
import time
import argparse

from mpi4py import MPI
from gaft import GAEngine
from gaft.analysis.fitness_store import FitnessStore
from gaft.analysis.console_output import ConsoleOutput
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from gaft.components import Population, DecimalIndividual
from gaft.operators import RouletteWheelSelection, UniformCrossover, FlipBitMutation

from calculate_acc import cal_acc
print('GENE algorithm to find best architecture')

parser = argparse.ArgumentParser(description='search for network architecture')
parser.add_argument('-gpu','--gpu_nums',required=True, type=int, help='number of your GPU')
args = parser.parse_args()


# Distribute task on multi-gpu
comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
GPU_ID = rank % args.gpu_nums
node_name = MPI.Get_processor_name() # get the name of the node
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
time.sleep(rank * 5)
print("node name: {}, GPU_ID: {}".format(node_name, GPU_ID))

# define population
indv_template = DecimalIndividual(ranges=[(100, 120), (60, 80)], eps=1)
population = Population(indv_template=indv_template, size=6)
population.init()

# Create genetic operators
selection = RouletteWheelSelection()
crossover = UniformCrossover(pc=0.8, pe=0.5)
mutation = FlipBitMutation(pm=0.1)

# Create genetic algorithm engine to run optimization
engine = GAEngine(population=population, selection=selection, \
                  crossover=crossover, mutation=mutation, \
                  analysis=[FitnessStore, ConsoleOutput])

@engine.fitness_register

def fitness(indv):
    fc1_channel, fc2_channel = indv.solution
    best_acc = cal_acc(fc1_channel, fc2_channel)
    print('fc1_channel: {} --- fc2_channel: {} ---best_accuracy {}'.format(fc1_channel, fc2_channel, best_acc))
    return float(best_acc.mean())

engine.run(ng=5)