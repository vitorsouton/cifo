from random import choice
from operator import attrgetter

def tournament_sel(population, tour_size=5):
    tournament = [choice(population) for _ in range(tour_size)]
    if population.optim == 'max':
        return max(tournament, key=attrgetter('fitness'))

    if population.optim == 'min':
        return min(tournament, key=attrgetter('fitness'))
