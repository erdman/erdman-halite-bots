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
parser.add_argument('--name',type=str, nargs='?', default='roibotV8a',help='Name of this version of the bot.')
parser.add_argument('--alpha', type=float, nargs='?', default=0.25, help='Alpha to use in exponential smoothing / discounting of ROI over distance.  Default is 0.25')  #Default was 0.1
parser.add_argument('--potential_degradation_step', type=float, nargs='?', default=0.5, help='Times friendly_distance ** 2 is potential degradation.  Default is 0.5')  #Default was 1.0
parser.add_argument('--enemy_ROI', type=float, nargs='?', default=-1.0, help='Amount of ROI to include for each enemy adjacent to an empty square.  Default is -1.0')
parser.add_argument('--hold_until', type=int, nargs='?', default=5, help='Hold square STILL until strength >= args.hold_until * production.  Default is 5.')
args = parser.parse_args()
logging.basicConfig(filename=args.name+'.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug(str(args))

myID, game_map = hlt.get_init()
if game_map.width == 20 or game_map.height == 20 or game_map.starting_player_count >= 4:
    args.hold_until = 6
    args.potential_degradation_step = 0.4
    assert args.hold_until == 6
    assert args.potential_degradation_step == 0.4
hlt.send_init(args.name)


def assign_move(square):
    potential, _, best_d, target = min((pf_map[neighbor] + (float('inf') if destinations[neighbor] + square.strength > 255 else 0), random.random(), direction, neighbor) for direction, neighbor in enumerate(game_map.neighbors(square))) # the random number breaks ties randomly
    staying_is_bad = (square.strength + square.production + destinations.get(square, 0)) > 255
    if potential == float('inf'):
        if staying_is_bad:
            # all 5 destinations are bad, choose least bad
            _, direction, target = min((square.strength + destinations.get(neighbor,0) + (square.production if direction == 0 else 0), direction, neighbor) for direction, neighbor in enumerate(game_map.neighbors(square, include_self=True)))
            return Move(square, direction)
        else:
            # OK to just stay
            return Move(square, STILL)
            
    if not staying_is_bad and any(Move(neighbor, best_d) in moves for neighbor in game_map.neighbors(square)):   #do not follow or mimic neighbors
        return Move(square, STILL)

    if not staying_is_bad and destinations[square] > 0:
        return Move(square, STILL)

    if staying_is_bad and any(destinations[neighbor] + square.strength < 256 for _, neighbor in originations[square]):
        return Move(square, min(originations[square], key = lambda tup: destinations[tup[1]])[0])

    if staying_is_bad:
        return Move(square, best_d)
      
    if target.owner != myID:  #an opponent -- actually can never be adjacent to opponent, it's always me or a "zero" owner square
        if (square.strength == 255) or (square.strength > target.strength):
            return Move(square, best_d)
    elif square.strength >= square.production * args.hold_until:   #target square is friendly, and we are strong enough to move
        return Move(square, best_d)

    return Move(square, STILL)

def initial_potential(square):
    # if empty, correlate with utility as an attacking square (if there are neighboring enemies)
    if square.owner == square.strength == 0:
        return sum(args.enemy_ROI for neighbor in game_map.neighbors(square) if neighbor.owner not in (0, myID))
    elif square.production == 0:
        return float('inf')
    else:
        return square.strength / square.production

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
            pf_map[square] = square_potential + args.potential_degradation_step * friendly_distance ** 2
            for neighbor in game_map.neighbors(square):
                if neighbor.owner != myID:
                    neighbor_potential  = (1 - args.alpha) * square_potential + args.alpha * (float('inf') if not neighbor.production else neighbor.strength / neighbor.production)
                    heapq.heappush(frontier, (neighbor_potential, random.random(), neighbor_potential, friendly_distance, neighbor))
                else:
                    neighbor_potential  = square_potential + args.potential_degradation_step * (friendly_distance + 1) ** 2 
                    heapq.heappush(frontier, (neighbor_potential, random.random(), square_potential, friendly_distance + 1, neighbor))
    moves = set() #list()
    destinations = defaultdict(int)
    originations = defaultdict(list)
    for square in sorted((square for square in game_map if square.owner == myID and square.strength > 0), key=lambda x: x.strength, reverse=True):
        move = assign_move(square)
        moves.add(move)
        target = game_map.get_target(square, move.direction)
        destinations[target] += square.strength
        originations[target].append((hlt.opposite_cardinal(move.direction), square))
    hlt.send_frame(moves)
    
    logging.debug(str(turn) + ' :: ' + str(int(1000 * (time.time() - start_time))))
