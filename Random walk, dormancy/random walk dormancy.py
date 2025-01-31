from tqdm import tqdm
import time

import numpy as np
import random
import matplotlib.pyplot as plt
# Random walk simulation
#x_0, y_0, a_0 are the initial coordinates and state of the walker
#t is the time elapsed



def simulate_walk_end_time(t, d, x_0=None, a_0=0, x_1=None, a_1=1,s_1=1,s_0=1,kappa=1,rho=1):
    return len(simulate_walk_hits(t, d, -1, x_0, a_0, x_1, a_1,s_1,s_0,kappa,rho))
def simulate_walk_hits(t_max, d, max_hits, x_0=None, a_0=0, x_1=None, a_1=1,s_1=1,s_0=1,kappa=1,rho=1):
    """Simule une marche aléatoire avec dormance jusqu'à ce qu'un certain nombre de hits soit atteint
    ou que le temps maximum soit dépassé.
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

def simulate_different_times(end_times,repetitions, d, x_0=None, a_0=0, x_1=None, a_1=1,s_1=1,s_0=1,kappa=1,rho=1):
    hits_different_times = np.zeros((len(end_times), repetitions))
    total_iterations = sum(end_times) * repetitions
    with tqdm(total=total_iterations, desc="Progress") as pbar:
        for i in range(len(end_times)):
            for j in range(repetitions):
                hits_different_times[i, j] = simulate_walk_end_time(end_times[i], d, x_0, a_0, x_1, a_1,s_1,s_0,kappa,rho)
                pbar.update(end_times[i])
    return hits_different_times
def simulate_different_s_1(end_time,repetitions, d, x_0=None, a_0=0, x_1=None, a_1=1,s_1=np.ones(1),s_0=1,kappa=1,rho=1):
    hits_different_times = np.zeros((len(s_1), repetitions))
    total_iterations = len(s_1) * repetitions
    with tqdm(total=total_iterations, desc="Progress") as pbar:
        for i in range(len(s_1)):
            for j in range(repetitions):
                hits_different_times[i, j] = simulate_walk_end_time(end_time, d, x_0, a_0, x_1, a_1,s_1[i],s_0,kappa,rho)
                pbar.update(1)
    return hits_different_times
def simulate_time(end_time,repetitions, d, x_0=None, a_0=0, x_1=None, a_1=1,s_1=1,s_0=1,kappa=1,rho=1):
    return simulate_different_times([end_time],repetitions, d, x_0, a_0, x_1, a_1,s_1, s_0, kappa, rho)[0,:]

def __main__():
    # Paramètres par défaut
    s_1 = 1
    s_0 = 1
    kappa = 1
    rho = 1
    gamma = -1
    Lambda = 1
    exponent = Lambda / (Lambda - gamma)

    max_time=100000
    repetitions = 500
    d = int(input("Veuillez entrer la dimension d: "))  # Demander à l'utilisateur la dimension
    points = [1.5**i for i in range(1, 33)]
    y=np.zeros(len(points))
    z=np.zeros(len(points))
    start_time = time.time()  # Démarrer le chronomètre
    hits_different_times=simulate_different_times(points, repetitions, d, [0] * d, 0, [0] * d,
                                              1, s_1, s_0, kappa, rho)
    y=np.mean(hits_different_times,axis=1)
    z=np.mean(np.power(exponent, hits_different_times),axis=1)
    """print("En", d, "dimensions:")
    print("Pour le temps t=", max_time, ", le nombre attendu de hits de (0,0,1) est",
              np.mean(hits_different_times[:]), "avec un écart-type de", np.std(hits_different_times[:]))
    feynman_kac = np.power(exponent, hits_different_times)
    print("En", d, "dimensions:")
    print("Pour le temps t=", max_time, ", la formule de Feynman-Kac donne", np.mean(feynman_kac[:]),
              "avec un écart-type de", np.std(feynman_kac[:]))"""


    end_time = time.time()  # Arrêter le chronomètre
    print("Le code a pris", end_time - start_time, "secondes pour s'exécuter.")

    plt.plot(points, y, label='times hit (0,0,1)')

    # Ajouter des labels et un titre
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graphique de hit de (0,0,1) en fonction de s_0')

    # Ajouter une légende
    plt.legend()

    # Afficher le graphique
    plt.show()
    plt.plot(points, z, label='FK')

    # Ajouter des labels et un titre
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Graphique de FK formula en fonction de s_0')

    # Ajouter une légende
    plt.legend()

    # Afficher le graphique
    plt.show()

    # Vérifier que les tableaux ont la même taille
    if len(points) == len(y) == len(z):
        with open('output.txt', 'w') as f:
            for a, b, c in zip(points, y, z):
                f.write(f"{a}\t{b}\t{c}\n")
    else:
        print("Les tableaux n'ont pas la même taille.")
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
