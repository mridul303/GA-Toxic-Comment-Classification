import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
import re
import random
import sys
import warnings
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

#Replace path with your path

data_path = "D:\\Documents\\DataSets\\Toxic_Classification\\train.csv"
data_raw = pd.read_csv(data_path)
print("Number of rows in data =",data_raw.shape[0])
print("Number of columns in data =",data_raw.shape[1])
print(data_raw.columns.values.tolist())
print("\n")
data_raw.head()

#Checking the catefories in the dataset
categories = list(data_raw.columns.values)


"""
    We need to 'clean' the dataset before we can use it. Cleaning invlolves removing of stopwords, punctuations, special chars etc.
    After cleaning, we use stemming to make it easier to handle large comments
"""

data = data_raw
if not sys.warnoptions:
    warnings.simplefilter("ignore")
def cleanHtml(sentence):
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, ' ', str(sentence))
    return cleantext
def cleanPunc(sentence): #function to clean the word of any punctuation or special characters
    cleaned = re.sub(r'[?|!|\'|"|#]',r'',sentence)
    cleaned = re.sub(r'[.|,|)|(|\|/]',r' ',cleaned)
    cleaned = cleaned.strip()
    cleaned = cleaned.replace("\n"," ")
    return cleaned
def keepAlpha(sentence):
    alpha_sent = ""
    for word in sentence.split():
        alpha_word = re.sub('[^a-z A-Z]+', ' ', word)
        alpha_sent += alpha_word
        alpha_sent += " "
    alpha_sent = alpha_sent.strip()
    return alpha_sent
data['comment_text'] = data['comment_text'].str.lower()
data['comment_text'] = data['comment_text'].apply(cleanHtml)
data['comment_text'] = data['comment_text'].apply(cleanPunc)
data['comment_text'] = data['comment_text'].apply(keepAlpha)
stop_words = set(stopwords.words('english'))
stop_words.update(['zero','one','two','three','four','five','six','seven','eight','nine','ten','may','also','across','among','beside','however','yet','within'])
re_stop_words = re.compile(r"\b(" + "|".join(stop_words) + ")\\W", re.I)
def removeStopWords(sentence):
    global re_stop_words
    return re_stop_words.sub(" ", sentence)
data['comment_text'] = data['comment_text'].apply(removeStopWords)
stemmer = SnowballStemmer("english")
def stemming(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence
data['comment_text'] = data['comment_text'].apply(stemming)

'''
    We are dumping the cleaned text as an object
    In case we want to rerun the GA part, we wouldn't have to go trhough the above process again
'''
pickle.dump(data,open("dataObj","wb"))

data = pickle.load(open("dataObj","rb"))
train, test = train_test_split(data, random_state=42, test_size=0.30, shuffle=True)

 #applying ordered crossover   
def breed(parent1, parent2):
    child = []
    childP1 = []
    childP2 = []
    
    idx = range(2,len(parent1))
    geneA, geneB = random.sample(idx, 2)
    startGene = min(geneA, geneB)
    endGene = max(geneA, geneB)

    for i in range(startGene, endGene):
        childP1.append(parent1[i])
        
    childP2 = [item for item in parent2 if item not in childP1]

    child = childP1 + childP2
    return child 
    
#Running mutation over the pop
def mutatePopulation(population, mutationRate):
    mutatedPop = []
    
    for ind in range(0, len(population)):
        mutatedInd = mutate(population[ind], mutationRate)
        mutatedPop.append(mutatedInd)
    return mutatedPop

#Mutatting 1 individual    
def mutate(individual, mutationRate):
    for swapped in range(len(individual)):
        if(random.random() < mutationRate):
            
            c = range(2,len(individual))
            idx1,idx2 = random.sample(c, 2)
            individual[idx1],individual[idx2] = individual[idx2],individual[idx1]
    return individual

def get_Fittest(fitness,population):
    maxFit = 0
    i = len(population)-1
    while(i):
        if(fitness[maxFit] < fitness[i]):
            maxFit = i
        
        i-=1
    return maxFit

def tournament(fitness,population):
    parents = []
    k = 3
    newpop = []
    idx = get_Fittest(fitness,population)
    #Keeping the fittest in the new pop
    newpop.append(population[idx])
    length = len(population)
    #Selecting 3 random parents for torunament
    while(length):
        for i in range(k):
            parents.append(random.choice(population))
        newpop.append(breed(parents[0],parents[1]))
        length-=1
        parents.clear()
    return newpop

def calc_Fitness(train_d):

    vectorizer = TfidfVectorizer(strip_accents='unicode', analyzer='word', ngram_range=(1,3), norm='l2')
    
    x_train = vectorizer.fit_transform(train_d.comment_text)
    y_train = train_d.drop(labels = ['id','comment_text'], axis=1)
    x_test = vectorizer.transform(test.comment_text)
    y_test = test.drop(labels = ['id','comment_text'], axis=1)
    
    # using classifier chains
    from sklearn.multioutput import ClassifierChain
    from sklearn.linear_model import LogisticRegression 
    from sklearn.metrics import accuracy_score, hamming_loss, precision_score
    # initialize classifier chains multi-label classifier
    classifier = ClassifierChain(LogisticRegression())
    # Training logistic regression model on train data
    classifier.fit(x_train, y_train)
    # predict
    predictions = classifier.predict(x_test)
    # accuracy
    quality = (accuracy_score(y_test,predictions) + (1 - hamming_loss(y_test,predictions)) + precision_score(y_test,predictions,average='weighted'))/3
    return quality

def swap_random(col):
    idx = range(2,len(col))
    i1, i2 = random.sample(idx, 2)
    col[i1] , col[i2] = col[i2] , col[i1]
    return col

def initialPop():
    population = []
    col = ['id','comment_text','toxic','severe_toxic','obscene','threat','insult','identity_hate']
    population.append(col[:])
    for i in range(1,10):
        population.append(swap_random(col[:]))
    return population

def getFitness(population):
    fitnessResults = []
    for i in range(0,len(population)):
        col = train.reindex(columns = population[i])
        fitnessResults.append(calc_Fitness(col))
    return fitnessResults
        
def nextGeneration(currentGen, mutationRate):
    fitness = getFitness(currentGen)
    children = tournament(fitness, currentGen)
    nextGeneration = mutatePopulation(children, mutationRate)
    return nextGeneration,fitness

def main():
    generations = 10
    mutation_rate = 0.015       
    pop = initialPop()
    for i in range(generations):
        print("Generation : ",i+1)
        pop , fitness= nextGeneration(pop, mutation_rate)
        print("\nFittest in this Generation : ",fitness[get_Fittest(fitness,pop)])
    fit = get_Fittest(fitness,pop)
    print("\nFittest = ",fitness[fit])
    print("\nOptimal Label Ordering :\n ",pop[fit].columns.values.tolist())
    
main()
