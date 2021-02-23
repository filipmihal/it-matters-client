import cv2
import numpy as np
import time
import datetime
import requests
from mtcnn.mtcnn import MTCNN

# SERVER_URL = 'http://127.0.0.1:8080'
# API_KEY = 'n2r5u8x/A?D(G+KbPeShVmYp3s6v9y$B&E)H@McQfTjWnZr4t7w!z%C*F-JaNdRg'
# USER_ID = '9ce50074-7c9f-43d8-b4de-3298ee5c8b86'
SERVER_URL =''
API_KEY = ''
USER_ID = ''
BATCH_SIZE = 30
DEFAULT_PERIOD = 8
QUICK_PERIOD = 1
HIGH_FREQUENCY_PERIOD = 25

period = 8
detector = MTCNN()


def is_looking_at_screen_mtcnn(input_image):
    return  True if len(detector.detect_faces(input_image)) != 0 else False


def draw_rec_face(input_image):
    faces = detector.detect_faces(input_image)
    for result in faces:
        x, y, w, h = result['box']
        x1, y1 = x + w, y + h
        cv2.rectangle(img, (x, y), (x1, y1), (0, 0, 255), 2)


def send_data_to_server(api_data):
    response = requests.post(SERVER_URL + '/api/user/' + USER_ID + '/report',
                             headers={'Content-Type': 'application/json',
                                      'Authorization': 'Bearer {}'.format(API_KEY)},
                             json={
                                 "report_data": api_data
                             })
    return response.status_code


def save_img_to_folder(image, folder_name):
    face_time = datetime.datetime.now()
    cv2.imwrite(folder_name + '/' + str(face_time) + '.png', image)


batch = []
cap = cv2.VideoCapture(0)
# small hack in order to get real time pics
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
prev_state = False
break_buffer = 0
while True:
    time.sleep(period)
    # clear buffer
    cap.grab()
    _, img = cap.read()
    is_watching = is_looking_at_screen_mtcnn(img)
    if is_watching:
        if not prev_state:
            print('I found a face, returning back to default period')
            period = DEFAULT_PERIOD
        break_buffer = 0
    else:
        break_buffer += period
        if prev_state:
            period = QUICK_PERIOD
            print('Increasing checking frequency')
        elif break_buffer < HIGH_FREQUENCY_PERIOD:
            print('saving picture ...')
            save_img_to_folder(img, 'no_face')
        elif break_buffer >= HIGH_FREQUENCY_PERIOD and period != DEFAULT_PERIOD:
            print('The break has reached the high frequency period. Slowing down ...')
            period = DEFAULT_PERIOD
    print('is looking at screen: ', is_watching)
    recorded_at = datetime.datetime.now().astimezone(datetime.timezone.utc)
    batch.append({
                'is_looking_at_screen': is_watching,
                'period': period,
                'recorded_at': recorded_at.isoformat()
    })
    if len(batch) >= BATCH_SIZE:
        print(send_data_to_server(batch))
        batch = []
    prev_state = is_watching
