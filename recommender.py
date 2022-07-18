__author__      = "Mitali Sahoo"

import numpy as np
import pandas as pd
import pdb
from numpy.linalg import norm
from numpy import dot
import math
from audioop import avg
from cmath import nan
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation, cosine, squareform, pdist
import ipywidgets as widgets
from IPython.display import display, clear_output
from sklearn.metrics import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt,ceil, floor
import sys, os, pdb
from contextlib import contextmanager

global metric,n_movies
global n_user


def train_data():
    with open("train.txt") as train_file:
        cols=['userId','itemId','rating']
        df=pd.read_table('train.txt',sep=' ',header=None,names=cols)
        n_users=df.userId.max()
        n_movie=df.itemId.max()

        rating=[[0 for x in range(n_movie)] for y in range(n_users)] 
        for  row in df.itertuples():
            rating[row[1]-1][row[2]-1]=row[3]
    return rating


def write_data(data, file_name):
    file = []
    for user, movie, rating in data:
        line = str(user)+" "+str(movie)+" "+str(rating)+"\n"
        file.append(line)
    fopen = open(file_name.replace("test","result"), 'w')
    fopen.writelines(file)


def test_data(file_name):
    with open(file_name) as train_file:
        data = [list(map(int, line.split())) for line in train_file]
    return list(data)  

def test_df(file_name):
    cols=['userId','itemId','rating']
    df=pd.read_table(file_name ,sep=' ',header=None,names=cols)
    n_users=df.userId.max()
    n_movie=df.itemId.max()

    rating=[[0 for x in range(n_movie)] for y in range(n_users)] 
    for  row in df.itertuples():
        rating[row[1]-1][row[2]-1]=row[3]
    return rating

#### Pearson Correlation Similarity ########################################

def pearson_correlation(a, b):
    if len(a) == 0 or len(b) == 0:
        return 0.0
    a -= np.mean(a)
    b -= np.mean(b)
    
    num = np.dot(a, b)
    den = norm(a)*norm(b)
    if den==0:
        return 0.0
    return num/den
    
def pearson_weighted_average(weight, rating):
    if np.sum(weight) == 0:
        return 0
    return np.sum(weight*rating)/np.sum(np.absolute(weight))

def weighted_average(weight, rating):
    if np.sum(weight) == 0:
        return 0

    return np.sum(np.array(weight)*np.array(rating))/np.sum(weight)

def user_row(data, size):
    temp = [0] * size
    for d in data:
        if d[1] >0:
            temp[d[0]-1] = d[1]
    return temp

def rounding(val):
    if val == 0:
        return 3
    elif val < 1:
        return 1
    elif val > 5:
        return 5
    else:
        return round(val)


def rounding_cosine(val,trainData,movie):
    if val == 0:
        #return 3
        # print(round(return_averagerate_movie(trainData,movie)))
        return round(return_averagerate_movie(trainData,movie))
    elif val < 1:
        return 1
    elif val > 5:
        return 5
    else:
        return round(val)

def movies_to_rate_by_user(testData, userid):
    return [row[1] for row in testData if row[0]==userid and row[2]==0]

def pearson(filename):
    trainData = train_data()
    testData = test_data(filename)
    testDf = test_df(filename)

    movies_count = 1000
    similarity_with_user = []
    test_users = list(set(j[0] for j in testData))
    result=[]
    for user in test_users:
        user_ratings = testDf[user-1]
        averageRating = np.mean([rating for rating in testDf[user-1] if rating !=0])
        similarity_with_user = []
        for peer in trainData:
            similarity_with_user.append(pearson_correlation(user_ratings, peer))
          
        for movie in movies_to_rate_by_user(testData, user):       
            coeff = np.array(similarity_with_user)
            peer_rating = [peer[movie-1] for peer in trainData]
            peer_rating_avg = np.mean(peer_rating) if len(peer_rating) > 0 else 0
            peer_rating = np.array([rating - peer_rating_avg for rating in peer_rating])
            result.append([user, movie, rounding(pearson_weighted_average(coeff, peer_rating) + averageRating)])
    write_data(result, filename)  


#### IUF Pearson #############################################################################

def pearsonIUF(filename):
    trainData = train_data()
    testData = test_data(filename)
    testDf = test_df(filename)
    iuf = []
    movies_count = 1000
    similarity_with_user = []
    test_users = list(set(j[0] for j in testData)) 
    result=[]

    m = len(trainData)
    train_t=pd.DataFrame(trainData).T.values.tolist()
    for x in train_t:
        mj=len([r for r in x if r>0])
        iuf.append(np.log(m/mj) if mj else 0.0)
    trainIUF = trainData * np.array(iuf)



    for user in test_users:
        user_ratings = testDf[user-1] 
        averageRating = np.mean([rating for rating in testDf[user-1] if rating !=0])
        similarity_with_user = [] 
        for peer in trainData:
            similarity_with_user.append(pearson_correlation(user_ratings, peer))
          
        for movie in movies_to_rate_by_user(testData, user):       
            coeff = np.array(similarity_with_user) 
            peer_rating = [peer[movie-1] for peer in trainIUF]
            peer_rating_avg = np.mean(peer_rating) if len(peer_rating) > 0 else 0
            peer_rating = np.array([rating - peer_rating_avg for rating in peer_rating])
            result.append([user, movie, rounding(pearson_weighted_average(coeff, peer_rating) + averageRating)])
    write_data(result, filename) 

######Case Amplification ########################################################################

def pearson_caseAmplification(filename):
    trainData = train_data()
    testData = test_data(filename)
    testDf = test_df(filename)
    p = 1.5
    movies_count = 1000
    similarity_with_user = []
    test_users = list(set(j[0] for j in testData))
    result=[]
    for user in test_users:
        user_ratings = testDf[user-1]
        averageRating = np.mean([rating for rating in testDf[user-1] if rating !=0])
        similarity_with_user = []
        for peer in trainData:
            similarity_with_user.append(pearson_correlation(user_ratings, peer))
        
        # print(similarity_with_user)
        # pdb.set_trace()
        a = np.array(similarity_with_user)
        # base = np.sign(a) * (np.abs(a)) ** (p)
        # similarity_with_user = similarity_with_user*pow(np.array(similarity_with_user), (p))
        similarity_with_user = np.sign(a) * (np.abs(a)) ** (p-1)
        
        k=0
        for movie in movies_to_rate_by_user(testData, user):       
            coeff = np.array(similarity_with_user)
            peer_rating = [peer[movie-1] for peer in trainData]
            peer_rating_avg = np.mean(peer_rating) if len(peer_rating) > 0 else 0
            peer_rating = np.array([rating - peer_rating_avg for rating in peer_rating])
            if math.isnan(pearson_weighted_average(coeff, peer_rating)): 
                result.append([user, movie,rounding(averageRating)])   
            else:
                result.append([user, movie,rounding(pearson_weighted_average(coeff, peer_rating) + averageRating)])

    write_data(result, filename) 

##### Cosine Similarity ##############################################################################################

def removingZeros1(a, b):
    # pdb.set_trace()
    removea = np.array([])
    removeb = np.array([])
    for first1, second2 in zip(a,b):
        if first1 and second2:
            removea = np.append(removea, first1)
            removeb = np.append(removeb, second2)
    # pdb.set_trace()
    return removea, removeb

def removingZeros(a,b):
    l = len(a)
    new_a, new_b = [], []
    for i in range(l):
        if a[i] * b[i] != 0:
        # if a[i] != 0 and b[i] != 0:
            new_a.append(a[i])
            new_b.append(b[i])
    return np.array(new_a), np.array(new_b)

def cosineCorrelation(a, b):
    a,b = removingZeros(a, b)
    # pdb.set_trace()
    if len(a) == 0 or len(b) == 0:
        return 0.0
    num = np.dot(a, b)
    den = norm(a)*norm(b)
    if den==0:
        return 0.0
    return num/den   

def cosine_similarity(filename):
    trainData = train_data()
    testData = test_data(filename)
    testDf = test_df(filename)

    movies_count = 1000
    similarity_with_user = []                      
    test_users = list(set(j[0] for j in testData))
    result=[]
    for user in test_users:
        user_ratings = testDf[user-1]
        averageRating = np.mean([rating for rating in testDf[user-1] if rating !=0])
        similarity_with_user = []
        for peer in trainData:
            # print(cosineCorrelation(user_ratings, peer))
            similarity_with_user.append(cosineCorrelation(user_ratings, peer))

        # print(similarity_with_user)

        for movie in movies_to_rate_by_user(testData, user):       
            coeff = np.array(similarity_with_user)  
            peer_rating = [peer[movie-1] for peer in trainData]
            coeff,peer_rating = removingZeros(coeff, peer_rating)
            # peer_rating_avg = np.mean(peer_rating) if len(peer_rating) > 0 else 0  
            # peer_rating = np.array([rating - peer_rating_avg for rating in peer_rating])

            result.append([user, movie, rounding_cosine(weighted_average(coeff, peer_rating),trainData,movie)]) 

    write_data(result, filename)  



####Own Algorithm#############################################################################################    

def get_similar_users(user, test_row, trainData, user_similarity_matrix):
    test_row_user, test_row_movie, test_row_rating = test_row[0], test_row[1], test_row[2]
  
    for train_row in trainData:
        if test_row_rating>0 and train_row[test_row_movie-1]>0 and test_row_rating>=train_row[test_row_movie-1]-1 and test_row_rating<=train_row[test_row_movie-1]+1:
            user_similarity_matrix.append(train_row)
    return user_similarity_matrix

def return_averagerate_movie(trainData,movie):
    sum_ratings, count = 0, 0
    for i in range(len(trainData)):
        if trainData[i][movie-1]:
            count += 1
            sum_ratings += trainData[i][movie-1]

    if not count:
        return 3
    return sum_ratings / count

def modified_algo(filename): 
    trainData = train_data()
    testData = test_data(filename)
    testDf = test_df(filename)

    user_similarity = [] 
    data = []
    user = None
    result=[]

    for test_row in testData:
        test_row_user, test_row_movie, test_row_rating = test_row[0], test_row[1], test_row[2]

        if user != test_row_user:
            user_similarity = []

        user_similarity = get_similar_users(user, test_row, trainData, user_similarity)
        if test_row_rating==0:
            count, total = 0, 0.0
            for k, user_row in enumerate(user_similarity):
                # if k > 280:
                    # break;
                if user_row[test_row_movie-1]>0:
                    total += user_row[test_row_movie-1]
                    count +=1
            if count:
                prediction = round(total/count)
            else:
                prediction = int(round(return_averagerate_movie(trainData,test_row_movie)))
                # print(prediction)
            result.append([test_row_user, test_row_movie, prediction])
        user=test_row_user                                     
    write_data(result, filename) 

#### Adjusted Cosine ##############################################################

def item_based_adjCosineMain(filename):
    k=200
    def adj_cosine(M):
        M_u = M.mean(axis=1)
        item_mean_subtracted = M - M_u[:, None]
        similarity_matrix = 1 - squareform(pdist(item_mean_subtracted.T, 'cosine'))
        return similarity_matrix

    def findksimilaritems_adjcos(item_id, ratings,  test_user_movies, k=k):
    
        zeroim = [1]*len(test_user_movies)
        test_user_movies_modified = [a - b for a, b in zip(test_user_movies, zeroim)]
        sim_matrix1 = adj_cosine(ratings)
    
        sim_matrix =pd.DataFrame(sim_matrix1)

        sim_movies = sim_matrix[item_id-1]  
        filtered_sim_movies = sim_movies.iloc[test_user_movies_modified]
        similarities = filtered_sim_movies.sort_values(ascending=False)[:k+1].values
        indices = filtered_sim_movies.sort_values(ascending=False)[:k+1].index


        if  math.isnan((M[item_id-1] !=0).sum()) :
            avg_item_rating = 3
        else: 
            avg_item_rating = sum(M[item_id-1])/  (M[item_id-1] !=0).sum() 
        

        return similarities ,indices, avg_item_rating

    def predict_itembased_adjcos(user_id, item_id, ratings, Mtest):
        def get_test_user_movies(user_id):

            all_movie_list = Mtest.loc[user_id - 1].tolist()
            test_movie_list = []
            for i,j in enumerate(all_movie_list) :
                if j != 0 :
                    test_movie_list.append(i+1)
            return (test_movie_list)

        def get_test_user_ratings(user_id):
            all_movie_list = Mtest.loc[user_id-1].tolist()
            return all_movie_list

        test_user_movies = get_test_user_movies(user_id)
        test_all_user_ratings = get_test_user_ratings(user_id)
        denom = [i for i in test_all_user_ratings if i != 0]
        mean = sum(test_all_user_ratings) // len(denom)
        similarities, indices, avg_item_rating =findksimilaritems_adjcos(item_id, ratings, test_user_movies) #similar users based on correlation coefficients

        prediction = avg_item_rating  
        sum_wt = np.sum(np.abs(similarities))

        product=1
        wtd_sum = 0
        for i in range(0, len(indices)):
            if indices[i]+1 == item_id:
                continue;
            else:
                product = (Mtest.iloc[user_id-1,indices[i]] - mean) * (similarities[i])
                wtd_sum = wtd_sum + product  

        if sum_wt == 0.0:
            prediction += 0                          
        else:
            prediction += int(round(wtd_sum/sum_wt))
        if prediction < 1:
            prediction = 1
        elif prediction >5:
            prediction = 5
        if math.isnan(prediction):
            prediction = 3
        else:
            prediction = int(round(prediction))
        return prediction


    cols=['userId','itemId','rating']
    df=pd.read_table('train.txt',sep=' ',header=None,names=cols)
    n_user=df.userId.max()
    n_movie=df.itemId.max()
    rating=np.zeros((n_user,n_movie))

    for  row in df.itertuples():
        rating[row[1]-1,row[2]-1]=row[3]
    M=pd.DataFrame(rating)
        
    dftest=pd.read_table(filename,sep=' ',header=None,names=cols)
    n_test_users=dftest.userId.max()
    n_test_movie=dftest.itemId.max()
    testrating=np.zeros((n_test_users,n_test_movie))


    for row in dftest.itertuples():
        testrating[row[1]-1,row[2]-1]=row[3]

    Mtest=pd.DataFrame(testrating)
    k=200
    metric='cosine' 

    original_file= open(filename,'r')
    res_file = open(filename.replace("test","result"), 'w')
    for line in original_file:
        element = line.split()
        user = int(element[0])
        movie = int(element[1])
        rate= int(element[2])
        if rate == 0:
            rating  = predict_itembased_adjcos(user,movie,M,Mtest)    
            # print(user,movie,rating)
            res_file.write('%s\t%s\t%s\n'% (user,movie,rating))
    res_file.close()
    original_file.close()      


###################################################################################
for test_file in ["test5.txt", "test10.txt", "test20.txt"]:
#uncomment any below function you want to run(run only one at a a time).
    # pearson(test_file)
    # pearsonIUF(test_file)    
    # pearson_caseAmplification(test_file)
    cosine_similarity(test_file)
    # modified_algo(test_file)
    #item_based_adjCosineMain(test_file)