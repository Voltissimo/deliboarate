from flask import Flask, request, jsonify
from engine import UCI

app = Flask(__name__)


@app.route('/position', methods=["GET"])
def get_position():
    fen = request.args.get('fen')
    moves = request.args.get('m')
    new_fen = UCI.position(fen, moves)
    success = True
    if new_fen == fen:
        success = False
    return jsonify({
        'success': success,
        'fen': new_fen
    })


@app.route('/go', methods=["GET"])
def get_best_move():
    fen = request.args.get('fen')
    depth = request.args.get('d', default=None)
    return jsonify({'move': UCI.go(fen, depth)})


if __name__ == '__main__':
    app.run()
