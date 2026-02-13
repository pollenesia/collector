
import argparse
import base64
import cv2
import json
import logging
import numpy as np
import os
import paho.mqtt.client as mqtt
import RPi.GPIO as gpio
import time

from datetime import datetime
from h5py import File
from io import BytesIO
from picamera2 import Picamera2
from PIL import Image
from RpiMotorLib import RpiMotorLib as rml

'''bash
rpicam-vid -o udp://192.168.1.64:8888 -t0 -v0 --flush=1 --framerate=10
ffplay udp://@:8888 -fflags nobuffer -flags low_delay -framedrop

rpicam-vid -o - -t0 -v0 --flush --inline --nopreview --framerate=30 --codec=yuv420 --width 1024 --height 1024 | nc -t4lp 8888
ffplay tcp://rpi1.local:8888 -f rawvideo -fflags nobuffer -pixel_format yuv420p -video_size 1024x1024 -framerate 30
'''

LOG_FORMAT = f"%(asctime)s [%(levelname)s] %(filename)s %(funcName)s(%(lineno)d): %(message)s"

THREAD_STEP_MM = 0.8
THREAD_LENGTH_MM = 30.0
THREAD_N = THREAD_LENGTH_MM / THREAD_STEP_MM
MAX_STEPS = int(512.0 * THREAD_N)

STEP_WAIT = 0.002
STEP_SIZE = 1000

DATA_CALLBACK = {}


def get_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger.addHandler(stream_handler)
    return logger


logger = get_logger(__name__)

PIN_STOP = 12
PIN_BREAK = 7
PIN_MOTOR1 = [18, 23, 24, 25]
PIN_MOTOR2 = [6, 13, 19, 26]
FORWARD = False
BACKWARD = True


def is_home() -> bool:
    return gpio.input(PIN_STOP) == False


def go_home(motor: rml.BYJMotor):
    if not is_home():
        motor.motor_run(PIN_MOTOR1, STEP_WAIT, STEP_SIZE,
                        BACKWARD, False, "full", 0.05)


def leave_home(motor: rml.BYJMotor):
    if is_home():
        motor.motor_run(PIN_MOTOR1, STEP_WAIT, STEP_SIZE,
                        FORWARD, False, "full", 0.05)


def go_to(din: dict, position: int) -> int:
    motor = din['motor_camera']
    is_home_fixed = din['is_home_fixed']
    position_step = din['position_step']

    if not is_home_fixed or position < 0 or position > MAX_STEPS:
        return 0

    delta = position - position_step
    delta_abs = abs(delta)
    step_size_abs = delta_abs if delta_abs < STEP_SIZE else STEP_SIZE
    if delta >= 0:
        direction = FORWARD
        step_size = +step_size_abs
    else:
        direction = BACKWARD
        step_size = -step_size_abs

    motor.motor_run(PIN_MOTOR1, STEP_WAIT, step_size_abs,
                    direction, False, "full", 0.05)

    return step_size


def go_steps(din: dict) -> dict:
    motor = din['motor_camera']
    is_home_fixed = din['is_home_fixed']
    position_step = din['position_step']
    steps = int(din['value'])

    if is_home_fixed:
        position_step1 = position_step + steps
        if position_step1 > MAX_STEPS:
            steps = MAX_STEPS - position_step
        elif position_step1 < 0:
            steps = -position_step

        if steps > 0:
            motor.motor_run(PIN_MOTOR1, STEP_WAIT, steps,
                            FORWARD, False, "full", 0.05)
        elif steps < 0:
            motor.motor_run(PIN_MOTOR1, STEP_WAIT, abs(steps),
                            BACKWARD, False, "full", 0.05)
        position_step += steps
    dout = {}
    dout['position_step'] = position_step
    return dout


def event_stop(channel: int):
    global DATA_CALLBACK
    motor = DATA_CALLBACK['motor_camera']
    is_home_fixed = DATA_CALLBACK['is_home_fixed']
    try:
        is_released = gpio.input(PIN_STOP)
        if is_home_fixed and is_released:
            logger.info('home leaved')
        elif is_home_fixed and not is_released:
            motor.motor_stop()
            logger.info('home reached')
        elif not is_home_fixed and is_released:
            motor.motor_stop()
            logger.info('limit switch released')
        else:
            motor.motor_stop()
            logger.info('home fixed')
    except Exception as e:
        logger.error(e)
        exit(-1)


def init_camera():
    camera = Picamera2()
    config = camera.create_still_configuration({'size': (1024, 1024)})
    camera.configure(config)
    camera.start()
    time.sleep(2.0)
    return camera


def get_image_file(camera: Picamera2, fname: str) -> dict[str, any]:
    data = camera.capture_file(fname)
    return data


def get_image(camera: Picamera2) -> Image:
    data = camera.capture_image()
    return data


def get_sharpness(fname: str):
    image = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    lv = cv2.Laplacian(image, cv2.CV_64F).var()
    return lv


def get_data_row(camera: Picamera2, fname: str, dkeys: list[str]):
    dimage = get_image(camera, fname)
    n = len(dkeys)
    row = np.ndarray(n)
    for i, key in enumerate(dkeys[:-3]):
        row[i] = dimage[key]
    row[-3] = get_sharpness(fname)
    row[-2] = position_step
    row[-1] = position_step / 512.0 * THREAD_STEP_MM
    return row


def release_break(release: bool):
    value = gpio.HIGH if release else gpio.LOW
    gpio.output(PIN_BREAK, value)
    logger.info(f'break release set to {release}')


def pass_tape(motor: rml.BYJMotor, range: int):
    release_break(True)
    logger.info(f'passing tape to {range} steps')
    motor.motor_run(PIN_MOTOR2, 0.01, range,
                    FORWARD, False, 'full', 0.05)
    logger.info('tape passed')
    release_break(False)


def init() -> dict:
    global DATA_CALLBACK
    data = {}

    logger.info('motors')
    gpio.setmode(gpio.BCM)
    gpio.setup(PIN_STOP, gpio.IN, gpio.PUD_UP)
    gpio.setup(PIN_BREAK, gpio.OUT, initial=gpio.LOW)
    gpio.add_event_detect(PIN_STOP, gpio.BOTH, event_stop, 200)

    data['motor_camera'] = rml.BYJMotor('camera', '28BYJ')
    data['motor_tape'] = rml.BYJMotor('tape', '28BYJ')

    logger.info('camera')
    data['camera'] = init_camera()

    logger.info('variables')
    data['is_home_fixed'] = False
    data['position_step'] = 0
    data['command'] = ''
    DATA_CALLBACK = data
    return data


def test_passing_with_brake():
    while True:
        release_break(True)
        time.sleep(1.0)
        pass_tape(64)
        time.sleep(1.0)
        release_break(False)
        time.sleep(3.0)


def on_message(client, userdata: dict, msg):
    msg_data = json.loads(msg.payload)
    logger.info(f"Received `{msg_data}` from `{msg.topic}` topic")
    command = msg_data['command']
    if command == 'release_break':
        release_break(msg_data['value'])
    elif command in ['go_home', 'find_home', 'go_end', 'get_image', 'go_steps', 'pass_tape', 'go_to']:
        userdata['command'] = command
        userdata['value'] = msg_data.get('value', 0)
    else:
        logger.warning(f'unknown command {msg_data}')


def init_mqtt(userdata) -> mqtt.Client:
    client = mqtt.Client(
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        protocol=mqtt.MQTTv5,
        transport='websockets',
        reconnect_on_failure=True
    )

    client.username_pw_set(username='user', password='12345678')

    client.connect(
        host='localhost',
        port=9001
    )

    client.subscribe('pollenesia/collector')
    client.on_message = on_message
    client.user_data_set(userdata)

    return client


def send_state(data: dict):
    mqtt_client: mqtt.Client
    mqtt_client = data['mqtt_client']
    d = {
        'is_home_fixed': data['is_home_fixed'],
        'position_step': data['position_step'],
        'command': data['command'],
    }
    payload = json.dumps(d)
    mqtt_client.publish('pollenesia/state', payload=payload)


def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     'host',
    #     default='localhost',
    #     help='MQTT host',
    #     type=str
    # )
    parser.add_argument('--output', help='output dir', default='.', type=str)

    try:
        options, _ = parser.parse_known_args()
        logger.info(vars(options))
    except Exception as e:
        logger.error(e)
        exit(0)

    options.output = os.path.expanduser(options.output)
    if not os.path.exists(options.output):
        os.makedirs(options.output)

    data = init()
    mqtt_client = init_mqtt(data)

    mqtt_client.loop_start()
    data['mqtt_client'] = mqtt_client
    motor_camera = data['motor_camera']
    # mqtt_client.loop_forever()

    # test_passing_with_brake()
    # is_home = True
    # go_end()

    dkeys = [
        'SensorTimestamp', 'ExposureTime',
        'FocusFoM', 'Lux', 'Sharpness', 'Position', 'Position_mm',
    ]

    mode = 'manual'
    m_step = 100
    m_pass_step = 100
    while True:
        if mode == 'auto':
            pass_tape(64)
            date = datetime.now()
            dirname = f'{(date.year % 100):02d}{date.month:02d}{date.day:02d}{date.hour:02d}{date.minute:02d}{date.second:02d}'
            logger.info(f'dirname: {dirname}')
            os.makedirs(dirname, exist_ok=True)
            init_camera()

            if leave_home():
                if go_home():
                    pass
                else:
                    logger.error('no home')
                    exit(-1)
            else:
                logger.error('limit switch stuck')
                exit(-1)

        img_data = np.ndarray((0, len(dkeys)))
        logger.info(img_data.shape)
        steps = 0
        i_image = 0
        while True:
            try:
                if mode == 'auto':
                    position_mm = steps / 512.0 * THREAD_STEP_MM
                    if position_mm < 15.0:
                        steps = go_steps(512)
                    elif position_mm >= 20.0:
                        logger.info(f'storing {dirname}/data.h5')
                        f = File(f'{dirname}/data.h5', 'w')
                        f['data'] = img_data
                        f['data'].attrs['header'] = dkeys
                        f.close()
                        dt = (datetime.now() - date).seconds
                        sleep_time = 600.0 - dt
                        logger.info(
                            f'cycle time is {dt}s, sleep {sleep_time}s')
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                        logger.info('finish')
                        break
                    else:
                        steps = go_steps(10)
                        # time.sleep(0.2)
                        fname = f'{dirname}/image{i_image:03d}.jpg'
                        row = get_data_row(data['camera'], fname, dkeys)
                        img_data = np.vstack((img_data, row))
                        i_image += 1
                elif mode == 'manual':
                    command = data['command']
                    if command == 'find_home':
                        leave_home(motor_camera)
                        if not is_home():
                            data['is_home_fixed'] = False
                            data['command'] = 'go_home'
                        else:
                            data['position_step'] += STEP_SIZE
                        send_state(data)
                    elif command == 'go_home':
                        go_home(motor_camera)
                        if is_home():
                            data['position_step'] = 0
                            data['is_home_fixed'] = True
                            data['command'] = ''
                        else:
                            data['position_step'] -= STEP_SIZE
                        send_state(data)
                    elif command == 'go_to':
                        step_size = go_to(data, data['value'])
                        if step_size == 0:
                            data['command'] = ''
                        else:
                            data['position_step'] += step_size
                        send_state(data)
                    elif command == 'get_image':
                        image = get_image(data['camera'])
                        buffer = BytesIO()
                        image.save(buffer, format='webp')
                        b = buffer.getvalue()
                        s = base64.b64encode(b).decode('utf-8')
                        payload = f'data:image/webp;base64,{s}'
                        mqtt_client.publish('pollenesia/img', payload=payload)
                        data['command'] = ''
                    elif command == 'go_steps':
                        d = go_steps(data)
                        data.update(d)
                        data['command'] = ''
                        data['value'] = 0
                    elif command == 'pass_tape':
                        pass_tape(data['motor_tape'], m_pass_step)
                        data['command'] = ''
                    else:
                        send_state(data)
                        time.sleep(0.2)
            except KeyboardInterrupt:
                logger.info('exit')

        exit(0)


if __name__ == '__main__':
    main()
