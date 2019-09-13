#!/usr/bin/env python
import eventlet
# has to be executed as early as possible to work
eventlet.monkey_patch()

import os
from flask import Flask, render_template, request, \
    copy_current_request_context

from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect, send

from kafka_channels_manager import KafkaChannelsManager

# Set this variable to "threading", "eventlet" or "gevent" to test the
# different async modes, or leave it set to None for the application to choose
# the best option based on installed packages.
# NOTE: only "threading" works for kafka setting!
#async_mode = None
#async_mode = 'threading'
async_mode = 'eventlet'

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
app.config['DEBUG'] = True
app.config['PORT'] = 5001
app.config['HOST'] = '0.0.0.0'
# connect:
#   ssh -L 19092:127.0.0.1:9092 gpu0
# use:
#   kafka://localhost:19092
app.config['KAFKA'] = 'kafka://localhost:9092'
app.config['WS_NAMESPACE'] = '/convqa'
app.config['KAFKA_TOPIC'] = 'convqa_out'
app.config['KAFKA_TOPIC_PROCESS_EXTERNAL'] = 'convqa_in'
app.config['ROOT'] = ''  # evtl use: '/backend'
if os.getenv('CONVQA_FRONTEND_SETTINGS', '') != '':
    # note: the path in CONVQA_FRONTEND_SETTINGS has to be relative to this script (frontend_only.py)
    print(f'load config from "{os.getenv("CONVQA_FRONTEND_SETTINGS")}"')
    app.config.from_envvar('CONVQA_FRONTEND_SETTINGS')
socketio = SocketIO(app, async_mode=async_mode,
                    client_manager=KafkaChannelsManager(url=app.config['KAFKA'], channel=app.config['KAFKA_TOPIC']))
#socketio = SocketIO(app, async_mode='threading', client_manager=KafkaManager(channel='convqa-out'))


@app.route('/')
def index():
    return render_template('chat_flink.html', async_mode=socketio.async_mode,
                           namespace=app.config['WS_NAMESPACE'], root=app.config['ROOT'])


def get_room():
    room = None
    for r in rooms():
        if r != request.sid:
            assert room is None, 'too many rooms available'
            room = r
    return room or request.sid


@socketio.on('ask', namespace=app.config['WS_NAMESPACE'])
def ask(message):
    if app.config['DEBUG']:
        print(f'message: {message}')
    room = get_room()
    m = {'data': message['user_input']}
    if 'username' in message:
        m['username'] = message['username']
    send(m, room=room)
    message['channel'] = app.config['KAFKA_TOPIC_PROCESS_EXTERNAL']
    emit('answer', message, room=room)


@socketio.on('send_message', namespace=app.config['WS_NAMESPACE'])
def send_message(message):
    m = {'data': message['data']}
    if 'username' in message:
        m['username'] = message['username']
    send(m, room=get_room())


@socketio.on('my_broadcast_event', namespace=app.config['WS_NAMESPACE'])
def broadcast_message(message):
    emit('system_message', {'data': message['data']}, broadcast=True)


@socketio.on('join', namespace=app.config['WS_NAMESPACE'])
def join(message):
    join_room(message['room'])
    emit('system_message', {'data': 'In rooms: ' + ', '.join(rooms())})


@socketio.on('enter', namespace=app.config['WS_NAMESPACE'])
def enter(message):
    left = []
    for room in list(rooms()):
        if room != request.sid:
            left.append(room)
            leave_room(room)
    join_room(message['room'])
    msg_return = 'joined room: ' + message['room']
    if len(left) > 0:
        msg_return = 'left rooms: ' + ', '.join(left) + '; ' + msg_return
    emit('system_message', {'data': msg_return})


@socketio.on('leave', namespace=app.config['WS_NAMESPACE'])
def leave():
    room = get_room()
    if room != request.sid:
        leave_room(room)
        emit('system_message', {'data': 'left room: ' + room})
    else:
        emit('system_message', {'data': 'you are in no room'})


@socketio.on('close_room', namespace=app.config['WS_NAMESPACE'])
def close(message):
    emit('system_message', {'data': 'Room ' + message['room'] + ' is closing.'}, room=message['room'])
    close_room(message['room'])


@socketio.on('my_room_event', namespace=app.config['WS_NAMESPACE'])
def send_room_message(message):
    emit('system_message', {'data': message['data']}, room=message['room'])


@socketio.on('disconnect_request', namespace=app.config['WS_NAMESPACE'])
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
    emit('system_message', {'data': 'Disconnected!'}, callback=can_disconnect)


@socketio.on('connect', namespace=app.config['WS_NAMESPACE'])
def connect():
    print('Client connected')
    emit('system_message', {'data': 'Connected'})


@socketio.on('disconnect', namespace=app.config['WS_NAMESPACE'])
def disconnect():
    print('Client disconnected', request.sid)


if __name__ == '__main__':
    print('NOTE: pre-requisite for this frontend is a Kafka queue connected to a Flink job that handles the background '
          'processing.\n The easiest way to achieve that is to start the default endpoint ("python web/endpoint.py ... ") '
          'and set up https://github.com/ArneBinder/queued-requester according to its readme.')
    socketio.run(app, debug=app.config["DEBUG"], host=app.config['HOST'], port=app.config["PORT"])
