from networkx.algorithms import community
import networkx as nx
G = nx.barbell_graph(5, 1)
communities_generator = community.girvan_newman(G)
top_level_communities = next(communities_generator)
next_level_communities = next(communities_generator)
print(sorted(map(sorted, next_level_communities)))