import pandas as pd
recipies = pd.read_csv('recipies.csv')
edges = pd.read_csv('edges_191120.csv')
nodes = pd.read_csv('nodes_191120.csv')
print(recipies.head())
print(nodes.head())
