# parsing command line arguments
import argparse
import csv
import os

def move_images(csv_file, new_path):
    data = csv.reader(open(csv_file, newline=''))

    lines = [l for l in data]
    print(len(lines))

    for line in lines:
        c = os.path.basename(line[0])
        l = os.path.basename(line[1])
        r = os.path.basename(line[2])

        line[0] = new_path + c
        line[1] = new_path + l
        line[2] = new_path + r

    writer = csv.writer(open('new.csv', 'w', newline=''))
    writer.writerows(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Script for setting new path to images in csv training set')
    parser.add_argument(
        'csv_file',
        type=str,
        nargs='?',
        default=None,
        help='Path to csv file'
    )
    parser.add_argument(
        'dist_path',
        type=str,
        nargs='?',
        default=None,
        help='New path for images'
    )

    args = parser.parse_args()

    if args.csv_file and args.dist_path:
        move_images(args.csv_file, args.dist_path)
    else:
        print("Specify path to csv file and new path for images")