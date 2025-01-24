import time

import numpy as np
import random
# Random walk simulation
#x_0, y_0, a_0 are the initial coordinates and state of the walker
#t is the time elapsed

# Paramètres par défaut
s_1 = 1
s_0 = 1
kappa = 1
rho = 1
gamma = 1

def simulate_walk_end_time(t, d, x_0=None, a_0=0, x_1=None, a_1=1):
    return len(simulate_walk_hits(t, d, -1, x_0, a_0, x_1, a_1))
def simulate_walk_hits(t_max, d, max_hits, x_0=None, a_0=0, x_1=None, a_1=1):
    """Simule une marche aléatoire avec dormance jusqu'à ce qu'un certain nombre de hits soit atteint ou que le temps maximum soit dépassé.
    Retourne une liste des temps entre les hits successifs."""
    ignore=False
    if x_0 is None:
        x_0 = [0] * d
    if x_1 is None:
        x_1 = [0] * d
    if max_hits==-1:
        ignore=True
    x = x_0[:]
    a = a_0
    hits = 0
    time_elapsed = 0
    hit_times = []
    old_hit=0

    while time_elapsed < t_max and (ignore or hits < max_hits):
        T = np.random.exponential(1)
        if a == 0:
            if all(coord == 0 for coord in x):
                T *= rho
                dx = random.choice([1, -1])
                axis = random.randint(0, d-1)
                x[axis] += dx
            else:
                T *= (rho + s_0)
                if np.random.uniform(0, 1) < s_0 / (rho + s_0):
                    a = 1
                else:
                    dx = random.choice([1, -1])
                    axis = random.randint(0, d-1)
                    x[axis] += dx
        else:
            if all(coord == 0 for coord in x):
                T *= (rho + kappa + s_1)
                if np.random.uniform(0, 1) < s_1 / (rho + kappa + s_0):
                    a = 0
                else:
                    dx = random.choice([1, -1])
                    axis = random.randint(0, d-1)
                    x[axis] += dx
            else:
                T *= (rho + kappa)
                dx = random.choice([1, -1])
                axis = random.randint(0, d-1)
                x[axis] += dx

        time_elapsed += T

        if x == x_1 and a == a_1:
            hits += 1
            hit_times.append(time_elapsed-old_hit)
            old_hit=time_elapsed
    return hit_times

def simulate_different_times(end_times,repetitions, d, x_0=None, a_0=0, x_1=None, a_1=1):
    hits_different_times = np.zeros((len(end_times), repetitions))
    for i in range(len(end_times)):
        for j in range(repetitions):
            hits_different_times[i, j] = simulate_walk_end_time(end_times[i], d)
    return hits_different_times

def __main__():
    times = [10, 100, 1000, 10000, 100000]
    repetitions = 1000
    for i in range(3):
        d = i + 1  # Dimension

        start_time = time.time()  # Démarrer le chronomètre
        hits_different_times = simulate_different_times(times, repetitions, d)
        end_time = time.time()  # Arrêter le chronomètre
        print("En", d, "dimensions:")
        for i in range(len(times)):
            print("Pour le temps t=", times[i], ", le nombre attendu de hits de (0,0,1) est",
                  np.mean(hits_different_times[i, :]), "avec un écart-type de", np.std(hits_different_times[i, :]))

        print("Le code a pris", end_time - start_time, "secondes pour s'exécuter.")

    
    """t_max=10000000
    d=1
    max_hits=3000
    start_time = time.time()  # Démarrer le chronomètre
    return_times = simulate_walk_hits(t_max, d, max_hits)
    end_time = time.time()  # Arrêter le chronomètre
    print("En ",d,"dimension, les temps de retour sont", np.mean(return_times), "avec un écart-type de", np.std(return_times))
    print("Le code a pris", end_time - start_time, "secondes pour s'exécuter.")"""
    
if __name__ == "__main__":
    __main__()
