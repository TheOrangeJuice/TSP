#test

import numpy as np
import pandas as pd
from sympy import sieve
import hvplot.pandas #custom
import colorcet as cc

cities = pd.read_csv("../input/cities.csv", index_col=['CityId'])
pnums = list(sieve.primerange(0, cities.shape[0]))
cities['isprime'] = cities.index.isin(pnums)

# show all points and density of primes
allpoints = cities.hvplot.scatter('X', 'Y',  width=380, height=350, datashade=True, 
                title='All Cities')
colors = list(reversed(cc.kbc))
primedensity = cities[cities.isprime].hvplot.hexbin(x='X', y='Y', width=420, height=350, 
                cmap=colors, title='Density of Prime Cities').options(size_index='Count', 
                min_scale=0.8, max_scale=0.95)
allpoints + primedensity

import hdbscan #custom

clusterer = hdbscan.HDBSCAN(min_cluster_size=600, min_samples=1)
clusterer.fit(cities[['X', 'Y']].values)
cities['clusts'] = clusterer.labels_

n = cities.clusts.max()
print("{} clusters".format(n))

custcolor = cc.rainbow[::8]
denses = cities.hvplot.scatter('X', 'Y',  by='clusts', size=5, width=500, height=450, 
                datashade=True, dynspread=True, cmap=custcolor)
denses

from sklearn.cluster import KMeans #utilizza algoritmo non supervisionato

misfits = cities[cities.clusts == -1].copy()
kmeans = KMeans(n_clusters=16, random_state=99, n_jobs=-1)
misfits['kclusts'] = kmeans.fit_predict(misfits[['X', 'Y']].values)
print("{} clusters".format(misfits.kclusts.max()))

sparses = misfits.hvplot.scatter('X', 'Y',  by='kclusts', width=500, height=450, legend=False,
                           datashade=True)
sparses

cities2 = cities.join(misfits.kclusts)
nextclust = cities.clusts.max()+1
cities2['clusts'] = np.where(cities2.clusts == -1, cities2.kclusts+nextclust, cities2.clusts)
cities2.drop('kclusts', axis=1, inplace=True)

centers = cities2.groupby('clusts')['X', 'Y'].agg('mean').reset_index()

def plot_it(df, dotsize, dotcolor, dotalpha):
    p = df.hvplot.scatter('X', 'Y', size=dotsize, xlim=(0,5100), ylim=(0,3400), width=500,
            height=450, hover_cols=['clusts'], color=dotcolor, alpha=dotalpha)
    return p

dpts = plot_it(centers[centers.clusts <= n], 30, 'darkblue', 1)
spts = plot_it(centers[centers.clusts > n], 40, 'blue', 0.2)
npole = plot_it(cities2.loc[[0]], 100, 'red', 1)
dpts*spts*npole

#%% imports
from scipy.spatial.distance import pdist, squareform
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2

#%% functions
def create_mat(df):
    print("building matrix")
    mat = pdist(locations)
    return squareform(mat)

def create_distance_callback(dist_matrix):
    def distance_callback(from_node, to_node):
      return int(dist_matrix[from_node][to_node])
    return distance_callback

def optimize(df):     
    tsp_size = len(locations) #params
    num_routes = 1
    mat = create_mat(df)
    dist_callback = create_distance_callback(mat)
    search_parameters = pywrapcp.RoutingModel.DefaultSearchParameters()
    search_parameters.time_limit_ms = int(1000*60*numminutes)
    search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.SIMULATED_ANNEALING)
    
    routemodel = pywrapcp.RoutingModel(tsp_size, num_routes, depot)
    routemodel.SetArcCostEvaluatorOfAllVehicles(dist_callback)
    
    print("optimizing") 
    assignment = routemodel.SolveWithParameters(search_parameters)

    print("Solver status: ", routemodel.status())
    print("Total distance: " + str(assignment.ObjectiveValue()))
    return routemodel, assignment
    
def get_route(df):
    routemodel, assignment = optimize(df)
    route_number = 0
    node = routemodel.Start(route_number)
    start_node = node
    route = []
    while not routemodel.IsEnd(node):
        route.append(node) 
        node = assignment.Value(routemodel.NextVar(node))
    print("Nodes:{} \n".format(len(route)))
    return route

#%% parameters
numminutes = 0.5

#%% main
depot = 40
locations = centers[['X', 'Y']].values
segment = get_route(locations)

opoints = centers.loc[segment]
line_ = opoints.hvplot.line('X', 'Y', xlim=(0,5100), ylim=(0,3400), color='green', width=500, height=450) 
denses*sparses*dpts*spts*npole*line_

