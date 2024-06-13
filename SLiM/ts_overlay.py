import msprime
import numpy as np
import tskit
import matplotlib
print('Hello world')
ts=tskit.load('overlay.trees').simplify()
mutated=msprime.sim_mutations(ts,rate=1e-7,random_seed=1,keep=True)
mutated.dump('final_overlaid.trees')
