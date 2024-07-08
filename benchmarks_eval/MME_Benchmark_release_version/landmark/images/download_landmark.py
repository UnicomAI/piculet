
import os
import argparse
import pandas as pd
import urllib.request
from tqdm import tqdm

def download_url_images(csv_file, output_folder,extension='.jpg'):
    df = pd.read_csv(csv_file)
    ids = df['id'].tolist()
    urls = df['url'].tolist()
    for i, url in tqdm(enumerate(urls), total=len(urls), desc='Downloading'):
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        save_path = output_folder + str(ids[i]) + extension
        try:
            urllib.request.urlretrieve(url, save_path)
        except:
            print("Failed to download the URL file:", url)
            print("Save_path:", save_path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Download images from URL in a CSV file.')
    parser.add_argument('--csv_file', help='Path to the CSV file')
    parser.add_argument('--output_folder', help='Output folder to save downloaded images')
    parser.add_argument('--extension', default='.jpg', help='File extension for downloaded images (default: .jpg)')
    args = parser.parse_args()

    # ide usage
    args.csv_file = './landmark200.csv'
    args.output_folder = './'
    args.extension = '.jpg'

    download_url_images(args.csv_file, args.output_folder, args.extension)


