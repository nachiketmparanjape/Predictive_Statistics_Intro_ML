'''This script demonstrates simulations of coin flipping'''
import random
import matplotlib.pyplot as plt
import numpy as np

# let's create a fair coin object that can be flipped:

class normal_var(object):
    '''this is a trial of a normal random variable'''
    choices = np.random.normal(0,0.1,100)
    last_result = None

    def normrun(self):
        '''call coin.normrun() to general a random normal outcome and return the result'''
        self.last_result = result = random.choice(self.choices)
        return result

# let's create some auxilliary functions to manipulate the coins:

def create_normal_outcomes(number):
    '''create a list of a number of coin objects'''
    return [normal_var() for _ in xrange(number)]

#def perform_normruns(runs):
#    '''side effect function, modifies object in place, returns None'''
#    for run in runs:
#        run.normrun()

def record_max(normal_objects):
    temp = []
    for i in normal_objects:
        temp.append(i.normrun())
    return max(temp)

def record_min(normal_objects):
    temp = []
    for i in normal_objects:
        temp.append(i.normrun())
    return min(temp)


def main():
    max_list = []
    runs = create_normal_outcomes(100)
    for i in xrange(100):
#        perform_normruns(runs)
        #print(count_heads(coins))
        max_list.append(record_max(runs))
    plt.hist(max_list,bins=100)
        

if __name__ == '__main__':
    main()
