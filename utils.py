import time
from base64 import b64encode
import cv2
import numpy as np
import pandas as pd
from scipy import stats
import os
import random
import linecache

from protos import benchmark_pb2 as pb2
import configs


def current_milli_time():
    return round(time.time() * 1000)


random.seed(33)
random_tracking_start = random.randint(0, 429)
current_tracking_number = random_tracking_start
audio_directory = "workloads/LibriSpeech-dev-clean"  # path to audio files dataset
sentiment_filename = "workloads/stanfordSentimentTreebank/datasetSentences.txt"  # Sentiment Analysis dataset
image_dataset_directory = "workloads/coco-dataset-val2017"  # image dataset (coco)
audio_alignment_dataset_dir = "workloads/Mozilla-dataset-2022-12-07/en"

audio_alignment_metadata_df = None
num_lines_in_sentiment_dataset = None
image_files = None
audio_files = None
script_dir = None

_SENTENCES = [
    "I love this camera, it works great!",
    "The battery life is terrible.",
    "Picture quality is amazing but autofocus is slow.",
    "Not worth the price.",
    "Absolutely fantastic, would buy again.",
    "It’s okay, but I expected more.",
    "The product stopped working after a week.",
    "Customer service was very helpful.",
    "The design is sleek and modern.",
    "Performance is fast and reliable."
]

def random_sentence():
    return random.choice(_SENTENCES)


def initial_workload_setup():
    global audio_files, image_files, num_lines_in_sentiment_dataset, audio_alignment_metadata_df, script_dir
    # get the path of the script and its directory
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)

    # Read the CSV file containing the metadata for the audio files and transcriptions
    audio_alignment_metadata_file = os.path.join(script_dir, audio_alignment_dataset_dir, "validated.tsv")
    audio_alignment_metadata_df = pd.read_csv(audio_alignment_metadata_file, delimiter="\t")

    # count the number of lines in the file
    num_lines_in_sentiment_dataset = sum(1 for line in open(os.path.join(script_dir, sentiment_filename)))

    # get a list of all files in the directory
    image_files = os.listdir(os.path.join(script_dir, image_dataset_directory))

    # filter out any non-files (e.g. subdirectories)
    image_files = [f for f in image_files if os.path.isfile(os.path.join(script_dir, image_dataset_directory, f))]

    # get a list of all audio dataset files and subdirectories in the directory
    all_audio_files = []
    for root, dirs, files in os.walk(os.path.join(script_dir, audio_directory)):
        for file in files:
            all_audio_files.append(os.path.join(root, file))

    # filter out any directories from the list of files
    audio_files = [f for f in all_audio_files if os.path.isfile(f)]


def get_random_image():
    # get a random file from the list
    # print(image_files)
    random_file = random.choice(image_files)
    im_file = os.path.join(script_dir, image_dataset_directory, random_file)
    im = cv2.imread(im_file)
    ret, buffer = cv2.imencode('.jpg', im)
    jpg_as_text = b64encode(buffer)
    base64_string = jpg_as_text.decode('utf-8')
    return base64_string


def get_random_sentiment_text():
    # choose a random line number
    random_line_num = random.randint(1, num_lines_in_sentiment_dataset)
    # get the random line
    random_line = linecache.getline(sentiment_filename, random_line_num)
    return random_line


def get_random_image_for_tracking():
    global current_tracking_number
    if current_tracking_number > 429:
        current_tracking_number = 1
    count_frame_id_digits = 1
    workload_input_path = configs.WORKLOAD_INPUT_PATH
    if current_tracking_number > 9:
        if current_tracking_number > 99:
            count_frame_id_digits = 3
        else:
            count_frame_id_digits = 2
    image_fl = workload_input_path + "0" * (6 - count_frame_id_digits) + str(current_tracking_number) + '.jpg'
    image_file = os.path.join(script_dir, image_fl)
    im = cv2.imread(image_file)
    ret, buffer = cv2.imencode('.jpg', im)
    jpg_as_text = b64encode(buffer)
    base64_string = jpg_as_text.decode('utf-8')
    current_tracking_number += 1
    return base64_string


def send_random_audio_in_chunks(send_defog=False):
    # Read the audio file and send it in chunks
    random_file = random.choice(audio_files)
    if send_defog:
        random_file = os.path.join(script_dir, 'workloads/audio/DeFog_b0430.wav')
    while '.txt' in random_file or '.DS_Store' in random_file or '2803-154328-0011.flac' in random_file:
        random_file = random.choice(audio_files)
    print(random_file)

    chunk_size = 8192  # Set an appropriate chunk size
    with open(random_file, 'rb') as audio_file:
        while True:
            timestamp = current_milli_time()
            data = audio_file.read(chunk_size)
            if not data:
                break
            yield pb2.SpeechToTextRequest(audio=data, request_time_ms=timestamp)


def send_random_audio_text_in_chunks():
    # Read the audio file and send it in chunks
    random_row = audio_alignment_metadata_df.sample(n=1).iloc[0]
    random_audio_file = os.path.join(script_dir, audio_alignment_dataset_dir, "clips", random_row["path"])
    transcription = random_row["sentence"]
    print(random_audio_file)
    print(transcription)

    chunk_size = 8192  # Set an appropriate chunk size
    with open(random_audio_file, 'rb') as audio_file:
        while True:
            timestamp = current_milli_time()
            data = audio_file.read(chunk_size)
            if not data:
                break
            yield pb2.AudioTextRequest(audio=data, text_input=transcription, request_time_ms=timestamp)


def get_avg_without_outlier(data_list):
    # return sum(data_list) / len(data_list)
    if np.std(data_list) == 0:
        return sum(data_list) / len(data_list)
    z = np.abs(stats.zscore(data_list))
    avg = 0
    count = 0
    for i in range(0, len(data_list)):
        if z[i] > -3 and z[i] < 3:
            avg += data_list[i]
            count += 1
        else:
            print("This is outlier: {}".format(data_list[i]))
    if count == 0:
        return 0
    return avg / count


def get_std_without_outlier(data_list):
    if len(data_list) == 0:
        return 0
    if np.std(data_list) == 0:
        return sum(data_list) / len(data_list)
    z = np.abs(stats.zscore(data_list))
    avg = 0
    count = 0
    list_without_outliers = []
    for i in range(0, len(data_list)):
        if z[i] > -3 and z[i] < 3:
            list_without_outliers.append(data_list[i])
            count += 1
        else:
            print("This is outlier: {}".format(data_list[i]))

    a = 1.0 * np.array(list_without_outliers)
    n = len(a)
    m, se = np.mean(a), stats.sem(a)
    h = se * stats.t.ppf((1 + 0.95) / 2., n - 1)
    return h