import numpy as np
from gudhi.simplex_tree import SimplexTree
import miniball
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_npy')
parser.add_argument('delaunay_txt')
parser.add_argument('--out', default='persistence.txt')
args = parser.parse_args()

print('Reading data')
data = np.load(args.data_npy)

tree = SimplexTree()

print('Reading simplices')
for line in tqdm(open(args.delaunay_txt, 'r').readlines()):
    simplex = list(map(int, line.strip().split(' ')))[1:]
    tree.insert(simplex)

print('Computing filtration')
for (simplex, _) in tqdm(tree.get_skeleton(tree.upper_bound_dimension())):
    coords = data[simplex, :]
    _, val = miniball.get_bounding_ball(coords)
    tree.assign_filtration(simplex, np.sqrt(val))

tree.make_filtration_non_decreasing()
tree.initialize_filtration()
tree.persistence()

print('Computing persistence')
tree.write_persistence_diagram(args.out)
