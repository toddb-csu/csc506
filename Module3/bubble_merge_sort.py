# Todd Bartoszkiewicz
# CSC506: Introduction to Data Structures and Algorithms
# Module 3: Critical Thinking Assignment
#
# This Python program will look at enhancing the efficiency of a hospital's patient
# records system by comparing the bubble sort and merge sort algorithms.
import random
import string
import time
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt


@dataclass
class Patient:
    patient_id: int
    last_name: str
    first_name: str
    admission_date: datetime


# Bubble Sort
def bubble_sort(bubble_sort_patient_records):
    number_of_patient_records = len(bubble_sort_patient_records)
    for i in range(number_of_patient_records):
        for j in range(0, number_of_patient_records - i - 1):
            if bubble_sort_patient_records[j].patient_id > patient_records[j + 1].patient_id:
                temp_bubble_sort_patient_record = bubble_sort_patient_records[j]
                bubble_sort_patient_records[j] = patient_records[j + 1]
                bubble_sort_patient_records[j + 1] = temp_bubble_sort_patient_record
    return bubble_sort_patient_records


# Merge Sort
def merge_sort(merge_sort_patient_records):
    if len(merge_sort_patient_records) <= 1:
        return merge_sort_patient_records
    mid_point = len(merge_sort_patient_records) // 2
    left_side = merge_sort(merge_sort_patient_records[:mid_point])
    right_side = merge_sort(merge_sort_patient_records[mid_point:])
    merged = []
    left_index = 0
    right_index = 0
    while left_index < len(left_side) and right_index < len(right_side):
        if left_side[left_index].patient_id <= right_side[right_index].patient_id:
            merged.append(left_side[left_index])
            left_index += 1
        else:
            merged.append(right_side[right_index])
            right_index += 1
    merged.extend(left_side[left_index:])
    merged.extend(right_side[right_index:])
    return merged


# Main function
if __name__ == "__main__":
    patient_record_counts = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    bubble_sort_times = []
    merge_sort_times = []

    for count in patient_record_counts:
        patient_records = [Patient(patient_id=random.randint(1, count),
                                   last_name=''.join(random.choices(string.ascii_uppercase, k=5)),
                                   first_name=''.join(random.choices(string.ascii_uppercase, k=3)),
                                   admission_date=datetime(2024, random.randint(1, 12), random.randint(1, 28)))
                           for i in range(count)]

        # Bubble Sort
        start_time = time.time()
        bubble_sort(patient_records.copy())
        end_time = time.time()
        bubble_sort_times.append(end_time - start_time)

        # Merge Sort
        start_time = time.time()
        merge_sort(patient_records.copy())
        end_time = time.time()
        merge_sort_times.append(end_time - start_time)

    plt.plot(patient_record_counts, bubble_sort_times, label='Bubble Sort')
    plt.plot(patient_record_counts, merge_sort_times, label='Merge Sort')
    plt.xlabel('Patient Record Count')
    plt.ylabel('Time (seconds)')
    plt.title('Sorting Algorithm Efficiency Comparison')
    plt.legend()
    plt.show()
