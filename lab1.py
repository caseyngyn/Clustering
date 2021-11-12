
import math
from typing import List, Tuple
import random
import numpy as np
import copy
import pandas as pd

def read_iris() -> Tuple[List[Tuple[float,str]], List[Tuple[int,List[float]]], List[List[int]]]:
    '''
    loser way
    '''
    actual_data = list()
    cluster = list()

    unlabelled = list()
    #iris_names = []
    i = 0
    with open("iris.data") as irisdata:
        for line in irisdata:
            if line.strip():
                line = line.rstrip()
                x = line.split(',')
                actual_data.append(x)
                y = x[0:4]
                y_convert = [float(i) for i in y]
                unlabelled.append(y_convert)
                #z = x[-1]
                cluster.append([i])
                i+=1
                #iris_names.append(z)
    #print(iris_datapoints)
    #print(iris_names)
    return actual_data,cluster,unlabelled

def read_glass():
    '''
    for chads only
    '''
    col_name = ['ID','RI','Na','Mg','Al','Si','K','Ca','Ba','Fe','GlassType']
    df = pd.read_csv('glass.data',names = col_name)
    #print(df)
    unlabelled = df.drop(['ID','GlassType'],axis=1)
    unlabelled= unlabelled.values.tolist()
    actual_data = df.drop('ID',axis=1)
    actual_data = actual_data.values.tolist()
    '''
    print('-----------------')
    for i in unlabelled:
        print(i)
    print('-----------------')
    
    print('actual',actual_data)
    '''  
    point = list(range(0,df.shape[0]))
    clustering = [[i] for i in point]
    ''' 
    print('-----------------')
    for i in clustering:
        print(i)
    print('-----------------')
    
    for i in range(len(unlabelled)):
        if len(unlabelled[i]) ==4:
            print(i)
            print('point', unlabelled[i])

    print(type(unlabelled[0]))
    '''
    return actual_data,clustering,unlabelled

def read_haberman():
    '''
    pandas is really cool
    '''
    col_name = ['Age','Year','num_nodes_detected','survival_stat']
    df = pd.read_csv('haberman.data',names = col_name)
    print(df)
    unlabelled = df.drop('survival_stat',axis=1)
    unlabelled= unlabelled.values.tolist()
    actual_data = df.values.tolist()
    '''
    print('-----------------')
    for i in unlabelled:
        print(i)
    print('-----------------')
    
    print('actual',actual_data)
    '''  
    point = list(range(0,df.shape[0]))
    clustering = [[i] for i in point]
    ''' 
    print('-----------------')
    for i in clustering:
        print(i)
    print('-----------------')
    
    for i in range(len(unlabelled)):
        if len(unlabelled[i]) ==4:
            print(i)
            print('point', unlabelled[i])

    print(type(unlabelled[0]))
    '''
    return actual_data,clustering,unlabelled


def euclidian_distance(x:List[float], y:List[float]) -> float:
    """
    parameter: 2 datapoints\n

    Iterate through data points add the summation of
    ((x_n)-(y_n))^2\n 

    returns: square root of summation
    """
    answer = 0
    ''' 
    print('x',len(x))
    if len(x) ==4:
        print(x)
    print('y',len(y))
    '''
    for i in range(len(x)):
        temp = (float(x[i]) - float(y[i]))**2
        answer =answer + temp
    return math.sqrt(answer)

def euclidian_distance_w_string(x:List[Tuple[float,str]], y:List[Tuple[float,str]]) -> float:
    """
    parameter: 2 datapoints for kmeans\n

    Iterate through data points add the summation of
    ((x_n)-(y_n))^2\n 

    returns: square root of summation
    """
    answer = 0
    for i in range(len(x)-1):
        temp = (x[i] - y[i])**2
        answer =answer + temp
    return math.sqrt(answer)

def get_sorted_distance(data:List[ Tuple[List[float]] ])-> list:
    """
    parameter: data: [int,[floats]]\n
    calc euclidian distance between every point.\n
    returns d -> list[index1, index2, distance]
    """
    d =list()
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            ed = euclidian_distance(data[i],data[j])
            d.append([i,j,ed])
    d.sort(key = lambda x:x[2])
    return d

def get_dist_matrix(data:List[List[float]]) -> List[List[float]]:
    """
    parameter: data[int, [floats]]\n
    return: nxn matrix of distances btwn data points
    """
    dist_matrix = list()

    for i in range(len(data)):
        row = list()
        for j in range(len(data)):
            distance = euclidian_distance(data[i],data[j])
            row.append(distance)
        dist_matrix.append(row)

    return dist_matrix


def single_linkage(data_labelled:List[Tuple[float,str]], cluster:List[List[int]], k:int,data_unlabelled:List[List[float]]) -> float:
    """
    parameters:
    - data_labelled: for hamming and to double check
    - data: without the string at the end and carries[ [index,[array]], [index,[array]], [index,[array]], etc..]
    - cluster: [ [index], [index] , [index] ] ---> [[index, index, index], [index, index], etc..] each point in its own cluster

    calc and store the distance btwn each pair of cluster\n
    while more than k cluster
        - let A,B be the two closest cluter
        - merge clusters B to A
        - delete B

    after cluster: run hamming distance and will show what cluster ran the best

    returns: hamming distance of single linkage
    """
    # single_linkage repeatedly takes the clustering which 
    # contain the two closest points and merges them, until k 
    # clustering remain
    #print("data",data)
    #distances is [index, index, ed]
    distances = get_sorted_distance(data_unlabelled)

    #we don't need to go through the whole distance array 
    #because once we seperate into k clusters we should stop algorithm
    while len(cluster)>k:
        #find 2 cluster1 andd cluster2 thrught shortest distance array
        for i in range(len(cluster)):
            for point in cluster[i]:
                if point == distances[0][0]:
                    cluster1 = i
                if point == distances[0][1]:
                    cluster2 = i

        #we don't need to re-calculate because distance between points will always be fixed
        #bc we just want minimum distance
        del(distances[0])
        if cluster1 == cluster2:
            continue
        
        #merge and remove cluster2 
        for point in cluster[cluster2]:
            cluster[cluster1].append(point)
        #print("post-cluster1",cluster[cluster1])
        del(cluster[cluster2])
    
    #debug
    '''
    for i in cluster:
        print(i)
    '''
    #TODO: make a truth cluster variable
    truth = truth_cluster(data_labelled)
    truth = convert_to_float(truth)
    ''' 
    for t in truth:
        print(t)
    print('---------------')'''
    actual_data_cluster = create_cluster(cluster,data_unlabelled)
    ''' 
    #debug
    for i in actual_data_cluster:
        print(i)
    ''' 
    print('-----------------SINGLE LINKAGE---------------------')
    print_clustering(actual_data_cluster)
    #TODO: hamming distance with actual_fata_cluster and the truth
    print('-------------TRUTH--------------------')
    for i in truth:
        print(i)
    hd = hamming(data_unlabelled,actual_data_cluster,truth)
    print('hd',hd)# hd 0.22857142857142856
    print('------------------------------------------------------')
    return hd


def average_linkage(data_labelled:List[Tuple[float,str]], data:List[List[float]], clustering:List[List[int]], k:int) -> float:
    """
    parameters:
    - data_labelled: for hamming and to double check
    - data: without the string at the end and carries[ [index,[array]], [index,[array]], [index,[array]], etc..]
    - cluster: [ [index], [index] , [index] ] ---> [[index, index, index], [index, index], etc..] each point in its own cluster

    - get nxn distance matrix
    while more than k cluster\n
        - iterate through cluster
            - find average distance btwn clusters
        - let A,B be the two closest cluter w/ min avg distance
        - add cluster A U B
        - remove clusters A and B
        - find distance between A U B and all other clusters

    after cluster: run hamming distance and will show what cluster ran the best

    returns: hamming distance of avg linkage
    """
    # average_linkage repeatedly merges the two clustering with
    # the smallest average distance until k clustering remain

    distances = get_dist_matrix(data)
    #print(clustering)
    while(len(clustering)> k):
        avg_min = float("inf")
        c1 = -1
        c2 = -1
        #enter the 1st list
        for cluster_i in range(len(clustering)):
            #i+1 to not get repeats
            for cluster_j in range(cluster_i+1,len(clustering)):
                #now inside list of int
                sum = 0.0
                cluster1 = clustering[cluster_i]
                cluster2 = clustering[cluster_j]
                #print("cluster1:",cluster1)
                #print("cluster2",cluster2)
                for x in cluster1:
                    for y in cluster2:
                        #print("x",x,"y",y,"distacne",distances[x][y])
                        sum += distances[x][y]
                #print('-----------------------------')
                avg = sum/(len(cluster1)*len(cluster2))

                if avg< avg_min:
                    #print("new avg_min,c1,c2: ",avg_min,c1,c2)
                    avg_min = avg
                    c1 = cluster_i
                    c2 = cluster_j

        #merge
        if (cluster1 == cluster2) or (c1 == -1):
            continue

        #print(clustering[c1])
        #print(clustering[c2])

        for point in clustering[c2]:
            clustering[c1].append(point)
        del(clustering[c2])   
    '''
    print('-----------')
    for i in clustering:
        print(i)
    print('-----------')
    '''

    truth = truth_cluster(data_labelled)
    truth = convert_to_float(truth)
    '''
    print('---------------')
    for t in truth:
        print(t)
    print('---------------')
    '''
    print('------------------AVERAGE--------------------')
    
    #debug
    actual_data_cluster = create_cluster(clustering,data)
    print_clustering(actual_data_cluster)
    print('-------------TRUTH--------------------')
    for i in truth:
        print(i)
    hd = hamming(data,actual_data_cluster,truth)
    print('hd',hd)# hd 0.10774049217002236
    print('--------------------------------------')
    return hd

def kmeanspp_centroids(data:List[List[float]], k:int) -> List[Tuple[List[float],str]]:
    '''
    probability equation: D(x)^2 / (summation(D(x)^2))
    x = points in data - centers we already chose
    D(x) denote the shortest distance from a data point to the closest center we have already chosen.

    we want to favor points farther away from initial chosen point. 
    with kmeanpp we will get a higher chance of points that are father away.
    compared to  
    '''
    random.seed()
    centroids = list()
    point = random.choice(data)
    centroids.append(point)
    #print(centroids)
    while len(centroids) < k:
        distances = list()
        for point in data:
            min_point_to_center_dist = math.inf
            for centroid in centroids:
                point_to_center = euclidian_distance(centroid,point)
                if point_to_center < min_point_to_center_dist:
                    min_point_to_center_dist = point_to_center
            distances.append(min_point_to_center_dist**2)
        d_sum = sum(distances)
        ''' 
        print('sum',d_sum)
        print('-------------b4----------------')
        for i in distances:
            print(i)
        '''
        distances =[i/d_sum for i in distances]
        '''
        print('--------------after------------')
        for i in distances:
            print(i)
        '''
        for i in range(len(distances)):
            if i>=1:
                distances[i] =distances[i]+ distances[i-1]
        '''
        print('--------------sum------------')            
        for i in distances:
            print(i)
        '''
        rand = random.random()
        for i in range(len(distances)):
            if rand < distances[i]:
                centroids.append(data[i])
                break
    '''
    print('--------------centroids------------')  
    print(centroids)
    '''
    return centroids

def llyod_centroids(data:List[List[float]], k:int) -> List[List[float]]:
    """
    Step 1: pick k random numbers\n
    returns list of centroids
    """

    random.seed()
    centroids = list()
    while len(centroids)<k:
        point = random.choice(data)
        centroids.append(point)
    centroids.sort()
    return centroids



def find_new_centroid(width, clustering:List[List[float]]) -> List[List[float]]:
    ''''
    calculate mean of each clustering per colimn
    '''
    #stripped_cluster =strip_end(clustering)
    '''
    print(' ------------------- STRIPPED ------------------- ')
    for i in stripped_cluster:
        print(i)
    print('------------------------------')
    '''
    centroids = list()
    for cluster in clustering:
        #get mean of whole cluster
        if len(cluster) != 0:
            cluster_mean =np.mean(cluster,axis=0)
            cluster_mean = np.round(cluster_mean,decimals = 5)
            cluster_mean = list(cluster_mean)
            centroids.append(cluster_mean)
        else:
            #if empty list
            centroids.append([0]*width)
    """
    print(' ------------------- NEW_CLUSTER_MEAN ------------------- ')
    for i in centroids:
        print(i)
    print('------------------------------')
    """ 

    return centroids



def kmeans(data:List[List[float]], k:int,method:str) -> List[float]:
    """
    Pick	k	random	points	(call	them	“centers”)\n	
    Until	convergence:	
        - Assign	each	point	to	its	closest	center.	This	gives	us	k clusters.		
	    - Compute	the	mean of	each	cluster.	
	    - Let	these	means	be	the	new	centers
    The	algorithm	converges	when	the	clusters	don’t	change	in	two	
    consecutive	iterations	
    check if old clustering is same to new clustering once we set new centroids
    """

    if method =='llyod':
        centroids = llyod_centroids(data,k)
    if method == 'kmeanspp':
        centroids = kmeanspp_centroids(data,k)

    condition = True

    width = len(data[0])
    #print('W',width)
    count = 1
    while condition:
        #print(count)
        clustering = list()
        for i in range(k):
            cluster = list()
            clustering.append(cluster)
        for point in data:
            #print('point',point)
            min_point_to_center = math.inf
            min_center = -1
            for j in range(len(centroids)):
                """
                finding distance between point and all cluster.
                """
                #print('pooint',point)
                #print('centroid',centroids[j])
                distance = euclidian_distance(centroids[j],point)

                if distance<min_point_to_center:
                    min_point_to_center = distance
                    min_center = j
            clustering[min_center].append(point)
        new_centroids = find_new_centroid(width,clustering)

        """
        print('-----------------------')
        print('------------------------------------CLuster---------')
        for x in clustering:
            print(x)
        
        print('------------------Centroids----------')
        print('old', centroids)
        print('new',new_centroids)
        print('--------------------------------------')
       """
        if new_centroids == centroids:
            #print('enter')
            condition = False
            break
        else:
            centroids = new_centroids
            #count +=1
    return clustering

def kmeans_cost(clustering,width):
    ''' 
    for i in clustering:
        print(i)
    '''
    temp = find_new_centroid(width,clustering)
    sum = 0
    for i in range(len(clustering)):
        #calc the centeroid: now a single point
        #add up 3 distances between point and centroid
        #add the sum of each cluster
        #print('cluster',clustering[i])
        for data in clustering[i]:
            #print('data',data)
            #print('temp',temp)
            distance = (euclidian_distance(data,temp[i]))**2
            sum += distance
    #print('sum',sum)
    return sum


def kmeans_run(data:List[List[float]], k:int, runs:int,data_labelled:List[Tuple[float,str]], method:str):
    """
    runs k means runs times
    takes best cluster
    find hamming distance compared to truth
    returns hamming distance
    """
    min_cost = math.inf
    #print(data)
    width = len(data[0])
    for i in range(runs):
        #taking mean of all centroids
        temp_clustering = kmeans(data,k,method)

        cost = kmeans_cost(temp_clustering,width)
        if cost<min_cost:
            best = temp_clustering
            min_cost = cost
    truth = truth_cluster(data_labelled)
    truth = convert_to_float(truth)
    hd = hamming(data,best,truth)
    print('------------',method,'-------------')
    print('-------------my algorithm------------')
    print_clustering(best)
    print('-------------TRUTH--------------------')
    for i in truth:
        print(i)
    print('hd',hd)
    print('-----------------------------------')


    return hd

def convert_to_float(truth_cluster):
    '''
    convert the truth cluster to float
    '''
    new_c = list()
    for clustering in truth_cluster:
        c = list()
        for cluster in clustering:
            l = list()
            for point in cluster:
                point = float(point)
                l.append(point)
            c.append(l)
        new_c.append(c)
    return new_c

            

def hamming(data,clustering,truth):
    '''
    Iterate 2 stacked points in the dataset
    Iterate through clustering
        Enter the cluster
        If point matches one of the 2 point stacked points
        Break
    Same process for truth
    If points are in same cluster for clustering but not in truth
        Add 1 to a
    If point  are not  in same clutter for clustering but same for truth
        Add 1 to b
    Return (a+b)/(n! /(2! *(n-2)!))   
    '''
    #find datapoint i and j in cluster and truth
    a = 0
    b = 0
    #print(data)
    for i in range(len(data)):
        for j in range(i+1,len(data)):
            c1 = -1
            c2 = -1
            c3 = -1
            c4 = -1
            for cluster_no in range(len(clustering)):
                for point in clustering[cluster_no]:
                    if point == data[i]:
                        c1 = cluster_no
                    if point == data[j]:
                        c2 = cluster_no
                    if c1 != -1 and c2 != -1:
                        #print('c1',c1)
                        #print('c2',c2)
                        break
            for cluster_no in range(len(truth)):
                for point in truth[cluster_no]:
                    if point == data[i]:
                        c3 = cluster_no
                    if point == data[j]:
                        c4 = cluster_no
                    if c3 != -1 and c4 != -1:
                        #print('c3',c3)
                        #print('c4',c4)
                        break
            #if 2 clusters are in the same clustering for my cluster algo 
            #but not together in the truth algo they disagree add 1 to a
            if (c1 == c2 and c3!= c4):
                a +=1
            #if 2 clusters are in the same clustering for my truth 
            #but not together in the cluster algo they disagree add 1 to b
            if (c1 != c2 and c3== c4):
                b+=1
    # (a+b)/(n! /(2! *(n-2)!))         
    return (a+b)/( math.factorial(len(data)) / (math.factorial(2)*math.factorial(len(data)-2)))

def truth_cluster(data):
    """
    parameter: cluster w/ label as data[-1]\n
    needed for HAMMING DISTANCE comparing algorithm with the truth cluster\n
    return actual clustered group
    """
    diff_type = list()
    for point in data:
        exists = False
        if len(diff_type) == 0:
            diff_type.append(point[-1])
            continue
        for c in diff_type:
            if point[-1] == c:
                exists = True
        if not exists:
            diff_type.append(point[-1])
    clustering = list()
    for cluster in diff_type:
        new_c = list()
        for point in data:
            if point[-1] == cluster:
                new_c.append(point)
        clustering.append(new_c)
    #labelled = copy.deepcopy(clustering)
    no_label = list()
    for data in clustering:
        for d in data:
            del(d[-1])
        no_label.append(data)
    return no_label

def create_cluster(cluster,data):
    """
    parameter: 
    - cluster: array of index numbers
    - data array of [index,[dataset]]

    match the cluster index number to data[i][0] and append data[i][1]
    to a new array

    return: clustered of datapoints
    """
    translated = list()
    for c in cluster:
        temp = list()
        for i in c:
            temp.append(data[i])
        translated.append(temp)
    return translated

def print_clustering(clustering):
    for cluster in clustering:
        print('length of cluster: ',len(cluster))
        print(cluster)

def main():
    #for glass you want 7 clusters
    #for haberman want 2
    #for iris want 3
    #bc in tuple it is now pass by reference
    label_data,cluster,unlabelled_data =read_haberman()
    #label_data,cluster,unlabelled_data =read_glass()
    #label_data,cluster,unlabelled_data = read_iris()#,name = read_iris()
    data_labelled = copy.deepcopy(label_data)
    data_labelled1 = copy.deepcopy(label_data)
    data_labelled2 = copy.deepcopy(label_data)
    clustering = copy.deepcopy(cluster)
    data_unlabelled = copy.deepcopy(unlabelled_data)
    data_unlabelled1 = copy.deepcopy(unlabelled_data)
    data_unlabelled2 = copy.deepcopy(unlabelled_data)
    ''' 
    print(id(label_data))
    print(id(data_labelled))
    print(id(data_labelled1))
    '''
    sl_hd = single_linkage(label_data,cluster,2,unlabelled_data)
   

    al_hd = average_linkage(data_labelled, data_unlabelled,clustering,2)
    


    kr_hs = kmeans_run(data_unlabelled1,2,100,data_labelled1,'llyod') #hd 0.12026845637583893

    krp_hs = kmeans_run(data_unlabelled2,2,100,data_labelled2,'kmeanspp')

    return 0

main()

