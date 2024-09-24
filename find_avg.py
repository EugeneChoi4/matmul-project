import csv

with open('timing-mine.csv', 'r') as file:
    reader = csv.DictReader(file)

    # Initialize variables for sum and count
    mflop_sum = 0
    count = 0

    # Iterate over the rows and sum the mflop values
    for row in reader:
        mflop_sum += float(row['mflop'])
        count += 1

    # Calculate the average
    average_gflop = mflop_sum / count / 1000 if count else 0

    print(f"Average gflop: {average_gflop}")
