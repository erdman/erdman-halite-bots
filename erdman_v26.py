#!/usr/bin/env python3

import hlt
from hlt import NORTH, EAST, SOUTH, WEST, STILL, Move, Square
import heapq
import random
import time
from collections import defaultdict
from itertools import chain, groupby
import logging
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--name',type=str, nargs='?', default='roibot',help='Name of this version of the bot.')  #defaults below correspond to "eleanor"
parser.add_argument('--alpha', type=float, nargs='?', default=0.10, help='Alpha to use in exponential smoothing / discounting of ROI over distance.  Default is 0.10')  #Default was 0.25
parser.add_argument('--potential_degradation_step', type=float, nargs='?', default=0.2, help='Times friendly_distance ** 2 is potential degradation.  Default is 0.2')  #Default was 0.5
parser.add_argument('--enemy_ROI', type=float, nargs='?', default=-0.5, help='Amount of ROI to include for each enemy adjacent to an empty square.  Default is -0.5')   #Default was -1.0
parser.add_argument('--fixed_hold',action='store_true', default=False, help='Hold_until stays fixed as specified, does not drop to 5 during exploration.')
parser.add_argument('--hold_until', type=int, nargs='?', default=7, help='In combat, hold square STILL until strength >= args.hold_until * production.  Default is 7.')
parser.add_argument('--int_max', type=float, nargs='?', default=0.45, help='Max proportion of interior pieces allowed to move.')
parser.add_argument('--int_min', type=float, nargs='?', default=0.01, help='Min proportion of interior pieces allowed to move.')
parser.add_argument('--enable_strategic_stilling', action='store_true', default=True, help='Enables strategic stilling behavior.')
parser.add_argument('--enable_red_green', action='store_true',default=True, help='Enables red-green trees for timing mining moves.')
args = parser.parse_args()
combat_hold_until = args.hold_until
logging.basicConfig(filename=args.name+'.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.debug(str(args))

myID, game_map = hlt.get_init()
hlt.send_init(args.name)

def assign_move(square):
    available_moves = sorted((degrade_potential(*pf_map[neighbor]) \
                                + 10000 * max(destinations[neighbor] + square.strength - 255, 0) \
                                + (1e7 if mining_remains and neighbor in wall else 0),
                                random.random(), direction, neighbor) for direction, neighbor in enumerate(game_map.neighbors(square))) # the random number breaks ties randomly
    potential, _, best_d, target = available_moves.pop(0)

    if square in greenlight:
        logging.debug(str(turn) + ' :: GREENlight:  ' + str(square) + ' went ' + str(best_d) + ', root ' + str(rootlookup[square]))
        return Move(square, best_d)

    stay_loss = max(0, min(255, square.strength + square.production) + destinations[square] - 255)

    if not stay_loss and (destinations[square] > 0 or square in redlight):     #safely meld with all oncoming
        return Move(square, STILL)

    dangerous_empties = {neighbor : min(255, sum(n2.strength for n2 in game_map.neighbors(neighbor, include_self=True) if n2.owner not in (0, myID)))
                            for neighbor in game_map.neighbors(square, n=2) \
                            if (neighbor.owner == neighbor.strength == 0 or neighbor.owner not in (0,myID)) \
                            and any(n2.owner not in (0, myID) and n2.strength > 0 for n2 in game_map.neighbors(neighbor, include_self=True)) \
                            and any(destinations[n2] > 0 for n2 in game_map.neighbors(neighbor, include_self=True))}

    if args.enable_strategic_stilling \
        and not stay_loss \
        and not dangerous_empties \
        and sum(1 for neighbor in game_map.neighbors(square) if neighbor.owner == neighbor.strength == 0) > 1 \
        and len(set(n2 for neighbor in game_map.neighbors(square) if neighbor.owner == neighbor.strength == 0 for n2 in game_map.neighbors(neighbor) if n2.owner not in (0,myID) and n2.strength >= 3 * n2.production)) > 1:
            return Move(square, STILL)   #strategic stilling ftw

    if any(neighbor in dangerous_empties for neighbor in game_map.neighbors(target, include_self=True)) \
        or (square.strength < args.hold_until * square.production and any(neighbor in dangerous_empties for neighbor in game_map.neighbors(square))):
        _, _, best_d, target = min((degrade_potential(*pf_map[neighbor]) \
                                + 10000 * max(destinations[neighbor] + square.strength - 255, 0) \
                                + 5000  * (sum(dangerous_empties.get(n2,0) for n2 in game_map.neighbors(neighbor, include_self = True)) if destinations[neighbor] == 0 else 0) \
                                + (1e7 if neighbor in wall else 0) \
                                + (100000 if neighbor.owner == 0 and square.strength <= neighbor.strength else 0), random.random(), direction, neighbor)
                                for direction, neighbor in enumerate(game_map.neighbors(square, include_self=True))) # the random number breaks ties randomly
        return Move(square, best_d)

    if stay_loss or potential > 9000:  #ok, now figure out which is worse
        if min(255, square.strength + square.production + destinations[square]) + min(255, destinations[target]) > min(255, destinations[square]) + min(255, destinations[target] + square.strength):
            # agreeing to losing squares by staying, bc better off than moving
            return Move(square, STILL)
        else:
            return Move(square, best_d)

    if target.owner != myID:  #an opponent -- actually can never be adjacent to opponent, it's always me or a "zero" owner square
        if ((square.strength == 255) or (square.strength + destinations[target] > target.strength)) \
            and square.strength >= 2 * square.production \
            and (destinations[target] == 0 or square.strength >= args.hold_until * square.production):
            return Move(square, best_d)

    elif square.strength >= max(strength_hurdle, args.hold_until * square.production):   #target square is friendly, and we are strong enough to move
        return Move(square, best_d)

    return Move(square, STILL)

def initial_potential(square):
    # if empty, correlate with utility as an attacking square (if there are neighboring enemies)
    if square.owner == square.strength == 0:
        return sum(args.enemy_ROI for neighbor in game_map.neighbors(square) if neighbor.owner not in (0, myID))
    elif square.production == 0 or square.owner not in (0,myID):
        return float('inf')
    elif square in wall:
        return 100 * square.strength / square.production
    else:
        return square.strength / square.production

def degrade_potential(potential, distance):
    return potential + args.potential_degradation_step * distance ** 2

def walk_tree(d, level = 1):
    for key, sub_d in d.items():
        yield level, key
        yield from walk_tree(sub_d, level + 1)

seen_enemies = set([0,myID])
turn = -1
while True:
    game_map.get_frame()
    start_time = time.time()
    turn += 1
    moves = []
    #modify potential's such that wall-block gives bare scent if enemy is unseen ... what about
    hero_empties = set(square for square in game_map if square.owner == square.strength == 0 and any(neighbor.owner == myID for neighbor in game_map.neighbors(square)))
    if not args.fixed_hold:
        args.hold_until = combat_hold_until if hero_empties else 5
    seen_enemies.update(game_map.neighbors(empty) for empty in hero_empties)    # THIS LINE HAS A MAJOR BUG AND DOESN'T DO WHAT IT'S SUPPOSED TO DO ... SEE WRITEUP
    wall = set(square for square in game_map if square.owner == 0 and square.strength > 0 and any(neighbor.owner == myID for neighbor in game_map.neighbors(square)) and any(neighbor.owner not in seen_enemies or neighbor.owner == neighbor.strength == 0 for neighbor in game_map.neighbors(square)))
    mining_remains = any(square for square in game_map if square.owner == 0 and square.production > 0 and square not in wall and any(neighbor.owner == myID for neighbor in game_map.neighbors(square)))
    frontier = [(initial_potential(square), random.random(), initial_potential(square), 0, square) for square in game_map if square.owner != myID]
    pf_map = dict()
    heapq.heapify(frontier)
    while len(pf_map) < game_map.width * game_map.height:
        _, _, square_potential, friendly_distance, square = heapq.heappop(frontier)
        if square not in pf_map:
            pf_map[square] = (square_potential, friendly_distance)
            for neighbor in game_map.neighbors(square):
                if neighbor in wall:
                    continue   #don't carve path through the wall, go around it
                elif neighbor.owner != myID:
                    neighbor_potential  = (1 - args.alpha) * square_potential + args.alpha * neighbor.strength / neighbor.production if neighbor.production and neighbor.owner == 0 else float('inf')
                    heapq.heappush(frontier, (neighbor_potential, random.random(), neighbor_potential, friendly_distance, neighbor))
                else:
                    neighbor_potential  = degrade_potential(square_potential, friendly_distance + 1)
                    heapq.heappush(frontier, (neighbor_potential, random.random(), square_potential, friendly_distance + 1, neighbor))

    trees = defaultdict(dict)
    edges = [(min((neighbor for neighbor in game_map.neighbors(square)), key=lambda x:degrade_potential(*pf_map[x])), square) for square in game_map if square.owner == myID]
    # Given a list of edges [parent, child], generate trees ... adapted from https://gist.github.com/aethanyc/8313640
    for parent, child in edges:
        trees[parent][child] = trees[child]
    parents, children = zip(*edges)
    roots = set(parents).difference(children)
    trees = {root: trees[root] for root in roots if root.owner == 0 and root.strength > 0}
    rootlookup = {node:root for root, tree in trees.items() for _, node in walk_tree(tree)}

    # create redlight and greenlight lists, which are "must still" and "must go" lists; some squares go in neither and are free to choose
    # for each tree, redlight the trunk that can't make it happen yet; greenlight the leaves that are ready to go
    redlight = set()
    greenlight = set()
    if args.enable_red_green:
        for root, tree in trees.items():
            accum_production = accum_strength = 0
            for distance, level in groupby(sorted(walk_tree(tree)), key = lambda x: x[0]):
                squares = list(list(zip(*level))[1])
                level_strength = sum(square.strength for square in squares)
                level_production = sum(square.production for square in squares)
                if accum_strength + accum_production > root.strength:   # accumulation is big enough, don't need to greenlight next square
                    break
                elif level_strength + accum_strength + accum_production > root.strength:
                    #remove unneeded squares from last layer
                    squares.sort(key = lambda x: x.strength, reverse=True)
                    while level_strength + accum_strength + accum_production - squares[-1].strength > root.strength:
                        level_strength -= squares.pop().strength
                    greenlight.update(squares)
                    break
                else:
                    accum_strength += level_strength + accum_production
                    accum_production += level_production
                    redlight.update(squares)

    moves = set()
    destinations = defaultdict(int)
    originations = defaultdict(list)
    interior_strengths = [square.strength for square in game_map if square.owner == myID and min(game_map.neighbors(square), key=lambda x: degrade_potential(*pf_map[x])).owner == myID]
    interior_strengths.sort(reverse=True)
    percentile = (1 - len(interior_strengths) / (50 * 50)) * (args.int_max - args.int_min) + args.int_min
    strength_hurdle = interior_strengths[int(len(interior_strengths) * percentile)] if interior_strengths else 0
    for square in sorted((square for square in game_map if square.owner == myID and square.strength > 0), key=lambda x: (x.strength, -pf_map[x][1]), reverse=True):  #when tied strength, move closest first
        move = assign_move(square)
        moves.add(move)
        target = game_map.get_target(square, move.direction)
        destinations[target] += square.strength + (square.production if move.direction == STILL else 0)
        originations[target].append((hlt.opposite_cardinal(move.direction), square))
    hlt.send_frame(moves)
    logging.debug(str(turn) + ' :: ' + str(int(1000 * (time.time() - start_time))))
