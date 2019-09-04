#!/usr/bin/env python
import eventlet
# has to be executed as early as possible to work
eventlet.monkey_patch()

from flask import Flask, render_template, session, request, \
    copy_current_request_context

from flask_socketio import SocketIO, emit, join_room, leave_room, \
    close_room, rooms, disconnect, send

from web.kafka_channels_manager import KafkaChannelsManager

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
app.config['PORT'] = 5000
app.config['HOST'] = '127.0.0.1'
#async_mode = 'threading' if app.config['DEBUG'] else 'eventlet'
NAMESPACE = '/convqa'
QUEUE = 'convqa_out'
QUEUE_EXT = 'convqa_in'
#socketio = SocketIO(app, async_mode=async_mode)
socketio = SocketIO(app, async_mode=async_mode, client_manager=KafkaChannelsManager(channel=QUEUE))
#socketio = SocketIO(app, async_mode='threading', client_manager=KafkaManager(channel='convqa-out'))


@app.route('/')
def index():
    return render_template('chat_flink.html', async_mode=socketio.async_mode, domain=app.config['HOST'],
                           port=app.config['PORT'], namespace=NAMESPACE)


def get_room():
    room = None
    for r in rooms():
        if r != request.sid:
            assert room is None, 'too many rooms available'
            room = r
    return room or request.sid


@socketio.on('ask', namespace=NAMESPACE)
def ask(message):
    #message_data = message['data']
    print(f'message: {message}')
    room = get_room()
    #user_room = request.sid
    #print(f'message_data: {message_data} (room: {user_room})')
    emit('question', {'question': message['user_input']}, room=room)
    message['channel'] = QUEUE_EXT
    emit('answer', message, room=room)


@socketio.on('send_message', namespace=NAMESPACE)
def send_message(message):
    send(message['data'], room=get_room())


@socketio.on('my_broadcast_event', namespace=NAMESPACE)
def broadcast_message(message):
    emit('system_message', {'data': message['data']}, broadcast=True)


@socketio.on('join', namespace=NAMESPACE)
def join(message):
    join_room(message['room'])
    emit('system_message', {'data': 'In rooms: ' + ', '.join(rooms())})


@socketio.on('enter', namespace=NAMESPACE)
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


@socketio.on('leave', namespace=NAMESPACE)
def leave():
    room = get_room()
    if room != request.sid:
        leave_room(room)
        emit('system_message', {'data': 'left room: ' + room})
    else:
        emit('system_message', {'data': 'you are in no room'})


@socketio.on('close_room', namespace=NAMESPACE)
def close(message):
    emit('system_message', {'data': 'Room ' + message['room'] + ' is closing.'}, room=message['room'])
    close_room(message['room'])


@socketio.on('my_room_event', namespace=NAMESPACE)
def send_room_message(message):
    emit('system_message', {'data': message['data']}, room=message['room'])


@socketio.on('disconnect_request', namespace=NAMESPACE)
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    # for this emit we use a callback function
    # when the callback function is invoked we know that the message has been
    # received and it is safe to disconnect
    emit('system_message', {'data': 'Disconnected!'}, callback=can_disconnect)


@socketio.on('connect', namespace=NAMESPACE)
def connect():
    print('Client connected')
    emit('system_message', {'data': 'Connected'})


@socketio.on('disconnect', namespace=NAMESPACE)
def disconnect():
    print('Client disconnected', request.sid)


if __name__ == '__main__':
    socketio.run(app, debug=app.config["DEBUG"], host=app.config['HOST'], port=app.config["PORT"])
