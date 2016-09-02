'''This script demonstrates simulations of coin flipping'''
import random
import matplotlib.pyplot as plt

# let's create a fair coin object that can be flipped:

class Coin(object):
    '''this is a simple fair coin, can be pseudorandomly flipped'''
    sides = ('heads', 'tails')
    last_result = None

    def flip(self):
        '''call coin.flip() to flip the coin and record it as the last result'''
        self.last_result = result = random.choice(self.sides)
        return result

# let's create some auxilliary functions to manipulate the coins:

def create_coins(number):
    '''create a list of a number of coin objects'''
    return [Coin() for _ in xrange(number)]

def flip_coins(coins):
    '''side effect function, modifies object in place, returns None'''
    for coin in coins:
        coin.flip()

def count_heads(flipped_coins):
    return sum(coin.last_result == 'heads' for coin in flipped_coins)

def count_tails(flipped_coins):
    return sum(coin.last_result == 'tails' for coin in flipped_coins)


def main():
    heads_list = []
    coins = create_coins(100)
    for i in xrange(10000):
        flip_coins(coins)
        #print(count_heads(coins))
        heads_list.append(count_heads(coins))
    plt.hist(heads_list, bins = 100)
        

if __name__ == '__main__':
    main()
