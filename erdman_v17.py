#!/usr/bin/env python3

import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import heapq
import random
import time
from collections import defaultdict
from itertools import chain
import logging
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str, nargs='?', default='roibot',help='Name of this version of the bot.')
parser.add_argument('--alpha', type=float, nargs='?', default=0.10, help='Alpha to use in exponential smoothing / discounting of ROI over distance.  Default is 0.10')  #Default was 0.25
parser.add_argument('--potential_degradation_step', type=float, nargs='?', default=0.2, help='Times friendly_distance ** 2 is potential degradation.  Default is 0.2')  #Default was 0.5
parser.add_argument('--enemy_ROI', type=float, nargs='?', default=-0.5, help='Amount of ROI to include for each enemy adjacent to an empty square.  Default is -0.5')   #Default was -1.0
parser.add_argument('--hold_until', type=int, nargs='?', default=5, help='Hold square STILL until strength >= args.hold_until * production.  Default is 5.')
parser.add_argument('--int_max', type=float, nargs='?', default=0.45, help='Max proportion of interior pieces allowed to move.')
parser.add_argument('--int_min', type=float, nargs='?', default=0.01, help='Min proportion of interior pieces allowed to move.')
args = parser.parse_args()
logging.basicConfig(filename=args.name+'.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug(str(args))

myID, game_map = hlt.get_init()
hlt.send_init(args.name)

def assign_move(square):
    available_moves = sorted((degrade_potential(*pf_map[neighbor]) + 10000 * max(destinations[neighbor] + square.strength - 255, 0), random.random(), direction, neighbor) for direction, neighbor in enumerate(game_map.neighbors(square))) # the random number breaks ties randomly
    potential, _, best_d, target = available_moves.pop(0)

    if potential > 9000:   # all 4 remaining destinations are bad, just go with the least bad; what about still?
        if destinations[square] < destinations[target]:
            best_d = STILL
        logging.debug(str(turn) + ' :: Least Bad!  ' + str(square) + ' went ' + str(best_d))
        return Move(square, best_d)

    staying_is_bad = (square.strength + destinations[square]) > 255
    if not staying_is_bad and destinations[square] > 0:     #safely meld with all oncoming
        return Move(square, STILL)

    dangerous_empties = set(neighbor for neighbor in game_map.neighbors(square, n=2) \
                            if neighbor.owner == neighbor.strength == 0 \
                            and any(n2.owner not in (0, myID) and n2.strength > 0 for n2 in game_map.neighbors(neighbor)) \
                            and any(destinations[n2] > 0 for n2 in game_map.neighbors(neighbor, include_self=True)))

    if dangerous_empties:
        # recalculating available_moves to consider still along with the 4 cardinals by same criteria
        available_moves = sorted((degrade_potential(*pf_map[neighbor]) + 10000 * max(destinations[neighbor] + square.strength - 255, 0), random.random(), direction, neighbor) for direction, neighbor in enumerate(game_map.neighbors(square, include_self=True))) # the random number breaks ties randomly
        while available_moves and any(neighbor in dangerous_empties for neighbor in game_map.neighbors(target, include_self=True)):
            _, _, best_d, target = available_moves.pop(0)
            logging.debug(str(turn) + ' :: Dangerous Empties!  ' + str(square) + ' trying to go ' + str(best_d))
        return Move(square, best_d)

    if staying_is_bad:
        while available_moves and target.owner != myID and square.strength + destinations[target] < min(target.strength, 254):
            potential, _, best_d, target = available_moves.pop(0)
        return Move(square, best_d)

    if target.owner != myID:  #an opponent -- actually can never be adjacent to opponent, it's always me or a "zero" owner square
        if ((square.strength == 255) or (square.strength + destinations[target] > target.strength)) and destinations[target] + square.strength <= 255 and square.strength >= 2 * square.production and (destinations[target] == 0 or square.strength >= args.hold_until * square.production):
            return Move(square, best_d)
    elif square.strength >= strength_hurdle and square.strength >= args.hold_until * square.production:   #target square is friendly, and we are strong enough to move
        return Move(square, best_d)

    return Move(square, STILL)

def initial_potential(square):
    # if empty, correlate with utility as an attacking square (if there are neighboring enemies)
    if square.owner == square.strength == 0:
        return sum(args.enemy_ROI for neighbor in game_map.neighbors(square) if neighbor.owner not in (0, myID))
    elif square.production == 0 or square.owner not in (0,myID):
        return float('inf')
    else:
        return square.strength / square.production

def degrade_potential(potential, distance):
    return potential + args.potential_degradation_step * distance ** 2

turn = 0
while True:
    start_time = time.time()
    turn += 1
    moves = []
    game_map.get_frame()
    frontier = [(initial_potential(square), random.random(), initial_potential(square), 0, square) for square in game_map if square.owner != myID]
    pf_map = dict()
    heapq.heapify(frontier)
    while len(pf_map) < game_map.width * game_map.height:
        _, _, square_potential, friendly_distance, square = heapq.heappop(frontier)
        if square not in pf_map:
            pf_map[square] = (square_potential, friendly_distance)
            for neighbor in game_map.neighbors(square):
                if neighbor.owner != myID:
                    neighbor_potential  = (1 - args.alpha) * square_potential + args.alpha * neighbor.strength / neighbor.production if neighbor.production and neighbor.owner == 0 else float('inf')
                    heapq.heappush(frontier, (neighbor_potential, random.random(), neighbor_potential, friendly_distance, neighbor))
                else:
                    neighbor_potential  = degrade_potential(square_potential, friendly_distance + 1)
                    heapq.heappush(frontier, (neighbor_potential, random.random(), square_potential, friendly_distance + 1, neighbor))
    moves = set()
    destinations = defaultdict(int)
    originations = defaultdict(list)
    interior_strengths = [square.strength for square in game_map if square.owner == myID and min(game_map.neighbors(square), key=lambda x: degrade_potential(*pf_map[x])).owner == myID]
    interior_strengths.sort(reverse=True)
    percentile = (1 - len(interior_strengths) / (50 * 50)) * (args.int_max - args.int_min) + args.int_min
    strength_hurdle = interior_strengths[int(len(interior_strengths) * percentile)] if interior_strengths else 0
    for square in sorted((square for square in game_map if square.owner == myID and square.strength > 0), key=lambda x: (x.strength, -pf_map[square][1]), reverse=True):  #when tied strength, move closest first
        move = assign_move(square)
        moves.add(move)
        target = game_map.get_target(square, move.direction)
        destinations[target] += square.strength
        originations[target].append((hlt.opposite_cardinal(move.direction), square))
    hlt.send_frame(moves)
    logging.debug(str(turn) + ' :: ' + str(int(1000 * (time.time() - start_time))))
