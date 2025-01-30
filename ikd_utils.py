# utils file for ikd_planner.py
# contains dijkstra and a* algos (2 different heuristics)
# edge weight calculation

import torch
import heapq
from collections import deque
import numpy as np


def edge_weight_calculation(trav_map, patch_sum):
    '''
    returns the edge weight when constructing the graph 

    trav_map : traversibility map from utils program
    patch_sum: boolean, whether to sum the traversibility footprint patch or use wheel differences

    returns: float
    '''
    edge_weight = 0

    if patch_sum:
        edge_weight = torch.sum(trav_map).cuda().unsqueeze(0).item()

    else:

        back_l = torch.mean(trav_map[ : , : 4, : 4]).item()
        back_r = torch.mean(trav_map[ : , : 4, trav_map.shape[2]-4 : ]).item()
        front_l = torch.mean(trav_map[ : , trav_map.shape[2]-4 : , : 4]).item()
        front_r = torch.mean(trav_map[ : , trav_map.shape[2]-4 : , trav_map.shape[2] - 4 : ]).item()
        edge_weight = abs(back_l - back_r) + abs(front_l - front_r) + abs(front_l - back_l) + abs(front_r + back_r)

    return edge_weight



def dij_map(map):
    """
    Creates a dictionary of poses that terminates at the end pose and provides
    the shortest path to the start node
    The output is used in conjuction with a helper function to provide the 
    shortest path in list form

    map: dictionary with graph
    {1x6 tuple-pose : {1x6 tuple-pose : integer-weight}}
    2 special keys: 'start', 'end'

    Returns 1x3 tuple
    prev : dictionary of poses
    start : 1x6 tuple which gives the starting pose
    end : 1x6 tuple which gives the ending pose
    """
    
    dist, prev = {}, {}

    heap = []
    
    for key in map:

        # deals with special keys
        if key in {'start', 'end'}:
            continue

        if key not in dist:
            dist[key] = float('inf')
            prev[key] = -1

        for v in map[key]:
            if v not in dist:
                dist[v] = float('inf')
                prev[v] = -1

    # -1 in prev is a stand in for undefined

    start, end = map['start'], map['end']

    dist[start] = 0
    prev[start] = 0

    heapq.heappush(heap, (0, start))

    while heap:
        # Only need weight in heap to give priority to easier paths 
        _, location = heapq.heappop(heap)

        if location == end:
            break
        
        if location not in map:
            continue

        for node in map[location]:
            node_weight = map[location][node]
            new_weight = dist[location] + node_weight

            if new_weight < dist[node]:
                prev[node] = location
                dist[node] = new_weight
                heapq.heappush(heap, (new_weight, node))

    return (prev, start, end)

def a_star_heuristic(location, end, weight, trad_heur):
    """
    Heuristic to multiply edge weight by distance
    then don't need to normalize edge weights
    Returns Edge weight multiplied by Euclidean distance, using as function h in A*
    location : 1x6 tuple-pose
    end: 1x6 tuple-pose that is end point
    weight: edge weight for particular node
    """

    if trad_heur:
        weight = 1

    return (((location[0] - end[0])**2 + (location[1] - end[1])**2)**(1/2)) * weight

def a_star_map(map, trad_heur):
    """
    map: dictionary with graph
    {1x6 tuple-pose : {1x6 tuple-pose : integer-weight}}
    2 special keys: 'start', 'end'
    """

    g_score, f_score, prev = {}, {}, {}

    heap = []

    for key in map:

        # deals with special keys
        if key in {'start', 'end'}:
            continue

        if key not in g_score:
            g_score[key] = float('inf')
            f_score[key] = float('inf')
            prev[key] = -1

        for v in map[key]:
            if v not in g_score:
                f_score[v] = float('inf')
                g_score[v] = float('inf')
                prev[v] = -1

    # -1 in prev is a stand in for undefined

    start, end = map['start'], map['end']

    g_score[start] = 0
    f_score[start] = a_star_heuristic(start, end, g_score[start], trad_heur)
    prev[start] = 0

    heapq.heappush(heap, (f_score[start], start))

    while heap:
        # Only need weight in heap to give priority to easier paths 
        _, location = heapq.heappop(heap)

        if location == end:
            break
        
        if location not in map:
            continue

        for node in map[location]:
            node_weight = map[location][node]
            new_weight = g_score[location] + node_weight

            if new_weight < g_score[node]:
                prev[node] = location
                g_score[node] = new_weight
                f_score[node] = new_weight + a_star_heuristic(node, end, g_score[node], trad_heur)
                heapq.heappush(heap, (f_score[node], node))

    return (prev, start, end)

def chosen_path(paths, target, source):
    """
    returns the shortest path from end to start

    paths : dictionary of poses 
    target : goal pose
    source : starting pose

    Returns numpy array
    """
    path_to_end = deque([target])

    curr = target

    while curr != source:
        nxt = paths[curr]
        # prevents starting pose from being included in path planning
        if nxt == source:
            break
        path_to_end.appendleft(nxt)
        curr = nxt

    return np.array(path_to_end)[:2]


def total_path_planner(map, type):
    '''
    allows single function to be called from file to return shortest path

    map : weighted graph built from function above
    type : type of planner called by ikd_planner.py
           1 - Dijkstra
           2 - A* Euclidian distance edge weight i.e trad_heur = True
           3 - A* Distance * edge weight i.e. trad_heur = False
    '''
    if type == 1:
        path, start, end = dij_map(map)
    elif type == 2:
        path, start, end = a_star_map(map, True)
    else:
        path, start, end = a_star_map(map, False)

    return chosen_path(path, end, start)