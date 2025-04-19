# Todd Bartoszkiewicz
# CSC506: Introduction to Data Structures and Algorithms
# Module 1: Critical Thinking Assignment
#
# This Python program is for a linear search.
import timeit


def linear_search(search_list, search_item):
    """
    Performs a linear search for a specific item in a lsit
    :param search_list: List of items to search
    :param search_item: Item to search for
    :return: int: Index of the item if found, otherwise return -1
    """

    for idx, item in enumerate(search_list):
        if item == search_item:
            return idx
    return -1


if __name__ == "__main__":
    # Setup code for using timeit
    SETUP_CODE = '''
from __main__ import linear_search
from random import randint
'''

    # Let's run through 3 examples with 3 lists of 100 numbers
    # In the first case, we'll pick a random number to find within the bounds of the list.
    # In the second case, we'll pick 0, since it is the first item in the list and the fastest use case.
    # In the third case, we'll pick 101, since it is not in the list and the longest use case.
    # We'll print the times for each case to run, so we can compare the run times.
    TEST_FIND_100_BEST_CASE = '''
mylist = [x for x in range(100)]
find = 0
result = linear_search(mylist, find)
print(f"Found {find} in list of {len(mylist)} items")
'''

    TEST_FIND_100 = '''
mylist = [x for x in range(100)]
find = randint(0, len(mylist))
result = linear_search(mylist, find)
print(f"Found {find} in list of {len(mylist)} items")
'''

    TEST_FIND_100_WORST_CASE = '''
mylist = [x for x in range(100)]
find = 101
result = linear_search(mylist, find)
print(f"Did not find {find} in list of {len(mylist)} items")
'''

    print(f"Random example of linear search in 100 item list")
    time_taken = timeit.timeit(stmt=TEST_FIND_100, setup=SETUP_CODE, number=1)
    print(f"100 items time taken: {time_taken} seconds")
    print(f"Best case of linear search in 100 item list")
    time_taken = timeit.timeit(stmt=TEST_FIND_100_BEST_CASE, setup=SETUP_CODE, number=1)
    print(f"100 items best case time taken: {time_taken} seconds")
    print(f"Worst case of linear search in 100 item list")
    time_taken = timeit.timeit(stmt=TEST_FIND_100_WORST_CASE, setup=SETUP_CODE, number=1)
    print(f"100 items worst case time taken: {time_taken} seconds\n")

    # Let's run through 3 more examples with 3 lists of 10,000 numbers
    # In the first case, we'll pick a random number to find within the bounds of the list.
    # In the second case, we'll pick 0, since it is the first item in the list and the fastest use case.
    # In the third case, we'll pick 10,001, since it is not in the list and the longest use case.
    # We'll print the times for each case to run, so we can compare the run times.
    TEST_FIND_10000_BEST_CASE = '''
mylist = [x for x in range(10000)]
find = 0
result = linear_search(mylist, find)
print(f"Found {find} in list of {len(mylist)} items")
'''

    TEST_FIND_10000 = '''
mylist = [x for x in range(10000)]
find = randint(0, len(mylist))
result = linear_search(mylist, find)
print(f"Found {find} in list of {len(mylist)} items")
'''

    TEST_FIND_10000_WORST_CASE = '''
mylist = [x for x in range(10000)]
find = 10001
result = linear_search(mylist, find)
print(f"Did not find {find} in list of {len(mylist)} items")
'''

    print(f"Random example of linear search in 10,000 item list")
    time_taken = timeit.timeit(stmt=TEST_FIND_10000, setup=SETUP_CODE, number=1)
    print(f"10,000 items time taken: {time_taken} seconds")
    print(f"Best case of linear search in 10,000 item list")
    time_taken = timeit.timeit(stmt=TEST_FIND_10000_BEST_CASE, setup=SETUP_CODE, number=1)
    print(f"10,000 items best case time taken: {time_taken} seconds")
    print(f"Worst case of linear search in 10,000 item list")
    time_taken = timeit.timeit(stmt=TEST_FIND_10000_WORST_CASE, setup=SETUP_CODE, number=1)
    print(f"10,000 items worst case time taken: {time_taken} seconds\n")

    # Let's run through 3 more examples with 3 lists of 1,000,000 numbers
    # In the first case, we'll pick a random number to find within the bounds of the list.
    # In the second case, we'll pick 0, since it is the first item in the list and the fastest use case.
    # In the third case, we'll pick 1,000,001, since it is not in the list and the longest use case.
    # We'll print the times for each case to run, so we can compare the run times.
    TEST_FIND_1000000_BEST_CASE = '''
mylist = [x for x in range(1000000)]
find = 0
result = linear_search(mylist, find)
print(f"Found {find} in list of {len(mylist)} items")
'''

    TEST_FIND_1000000 = '''
mylist = [x for x in range(1000000)]
find = randint(0, len(mylist))
result = linear_search(mylist, find)
print(f"Found {find} in list of {len(mylist)} items")
'''

    TEST_FIND_1000000_WORST_CASE = '''
mylist = [x for x in range(1000000)]
find = 1000001
result = linear_search(mylist, find)
print(f"Did not find {find} in list of {len(mylist)} items")
'''

    print(f"Random example of linear search in 1,000,000 item list")
    time_taken = timeit.timeit(stmt=TEST_FIND_1000000, setup=SETUP_CODE, number=1)
    print(f"1,000,000 items time taken: {time_taken} seconds")
    print(f"Best case of linear search in 1,000,000 item list")
    time_taken = timeit.timeit(stmt=TEST_FIND_1000000_BEST_CASE, setup=SETUP_CODE, number=1)
    print(f"1,000,000 items best case time taken: {time_taken} seconds")
    print(f"Worst case of linear search in 1,000,000 item list")
    time_taken = timeit.timeit(stmt=TEST_FIND_1000000_WORST_CASE, setup=SETUP_CODE, number=1)
    print(f"1,000,000 items worst case time taken: {time_taken} seconds\n")
