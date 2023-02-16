import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


G= nx.DiGraph() #creation of a dircted graph

"""
G.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11,12])
G.add_edges_from([(1, 4),(2, 5), (3, 6), (4, 7), (5 , 7), (5, 8), (5, 9), (6, 9), (7, 10), (8, 10), (9, 11), (11, 12)])
attrs = {1: {'task1': 6}, 2: {'task2': 23}, 3: {'task3': 25}, 4: {'task4': 16.5}, 
5: {'task5': 20}, 6: {'task6': 12}, 7: {'task7': 9}, 8: {'task8': 11}, 
9: {'task9': 14}, 10: {'task10': 8.5}, 11: {'task11': 17}, 12: {'task12': 13}}"""

"""example01
G.add_nodes_from([1,2,3,4,5,6])
G.add_edges_from([(1, 4),(2, 4), (2, 5), (3, 5), (4 , 6), (5, 6)])
attrs = {1: {'task1': 2}, 2: {'task2': 3}, 3: {'task3': 2}, 4: {'task4': 3}, 5: {'task5': 2}, 6: {'task6': 1}}
"""

"""
G.add_nodes_from([1,2,3,4,5,6,7,8,9,10])
G.add_edges_from([(1, 5),(2, 6), (3, 7), (4, 7), (7, 9), (5 , 8), (6, 8), (8, 10), (9, 10)])
attrs = {1: {'task1': 2}, 2: {'task2': 3}, 3: {'task3': 1}, 4: {'task4': 2}, 5: {'task5': 1}, 6: {'task6': 4}, 7: {'task7': 2}
, 8: {'task8': 3}, 9: {'task9': 5}, 10: {'task10': 2}}
nx.set_node_attributes(G,attrs)
"""
"""
#example 01
G.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11,12])
G.add_edges_from([(1, 3),(1, 4), (1, 2), (1, 8), (2, 7), (3 , 7), (4, 5), (5, 6), (6, 7)
, (7, 11), (8, 9), (9, 10), (10, 11), (11, 12)])
attrs = {1: {'task1': 4.5}, 2: {'task2': 4.0}, 3: {'task3': 3.0}, 4: {'task4': 6.0}, 5: {'task5': 7.0}, 6: {'task6': 3.0}, 7: {'task7': 7.0}
, 8: {'task8': 3.0}, 9: {'task9': 3.0}, 10: {'task10': 7.0}, 11: {'task11': 4.5}, 12: {'task12': 2.0}}
nx.set_node_attributes(G,attrs)
#---------------------------------
"""
"""
#example 02
G.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
G.add_edges_from([(1, 6),(1, 7), (2, 7), (2, 8), (2, 9), (3 , 9), (3, 10), (3, 16), (4, 11)
, (5, 11), (6, 20), (7, 12), (8, 12), (9, 15), (10, 20), (11, 13), (11, 14), (11, 16), (12, 15)
, (13, 17), (14, 18), (14, 19), (15, 20), (16, 20), (17, 19), (18, 20), (19, 20)])
attrs = {1: {'task1': 0.3}, 2: {'task2': 0.8}, 3: {'task3': 0.3}, 4: {'task4': 0.4}, 5: {'task5': 0.2}, 6: {'task6': 0.2}, 7: {'task7': 0.5}
, 8: {'task8': 0.5}, 9: {'task9': 0.3}, 10: {'task10': 0.2}, 11: {'task11': 0.3}, 12: {'task12': 0.3}, 13: {'task13': 0.67}, 14: {'task14': 0.2}
, 15: {'task15': 0.77}, 16: {'task16': 0.1}, 17: {'task17': 0.5}, 18: {'task18': 0.37}, 19: {'task19': 0.35}, 20: {'task20': 0.1}}
nx.set_node_attributes(G,attrs)
#---------------------------------
"""

#example 03
G.add_nodes_from([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25])
G.add_edges_from([(1, 3),(2, 8), (2, 9), (3, 9), (4, 7), (5, 6), (6, 10), (6, 11), (7, 8)
, (8, 12), (8, 13), (9, 19), (9, 20), (10, 13), (11, 15), (11, 18), (13, 14), (15, 16), (15, 17)
, (16, 24), (17, 23), (18, 23), (20, 21), (21, 22), (23, 25), (24, 25)])
attrs = {1: {'task1': 2.4}, 2: {'task2': 3.2}, 3: {'task3': 1.9}, 4: {'task4': 0.7}, 5: {'task5': 1.9}, 6: {'task6': 0.8}, 7: {'task7': 1.5}
, 8: {'task8': 2.2}, 9: {'task9': 0.4}, 10: {'task10': 0.9}, 11: {'task11': 1.4}, 12: {'task12': 2}, 13: {'task13': 1.3}, 14: {'task14': 0.9}
, 15: {'task15': 3.3}, 16: {'task16': 1.6}, 17: {'task17': 1.3}, 18: {'task18': 1.5}, 19: {'task19': 3.8}, 20: {'task20': 1.6}, 21: {'task21': 1.2}
, 22: {'task22': 2.5}, 23: {'task23': 2.5}, 24: {'task24': 2.4}, 25: {'task25': 2.2}}
nx.set_node_attributes(G,attrs)
#--------------------------------
