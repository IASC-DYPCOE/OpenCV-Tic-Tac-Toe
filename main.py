from flask import Flask, request, jsonify, session
import cv2
import numpy as np
import mediapipe as mp
import random
import math

app = Flask(__name__)
app.secret_key = "supersecretkey"  # Needed for session management

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    model_complexity=0,
)

CELL_SIZE = 200
HUMAN_PLAYER = 1
COMPUTER_PLAYER = -1


def init_game():
    session["board"] = np.zeros((3, 3), dtype=int).tolist()
    session["game_over"] = False
    session["winner"] = None
    session["difficulty"] = "medium"


@app.route("/reset_game", methods=["POST"])
def reset_game():
    init_game()
    return jsonify({"message": "Game reset successful"})


def check_winner(board):
    board = np.array(board)
    for i in range(3):
        if abs(sum(board[i, :])) == 3:
            return board[i, 0]
        if abs(sum(board[:, i])) == 3:
            return board[0, i]
    if abs(sum(np.diag(board))) == 3:
        return board[0, 0]
    if abs(sum(np.diag(np.fliplr(board)))) == 3:
        return board[0, 2]
    if np.count_nonzero(board) == 9:
        return 0
    return None


def get_empty_cells(board):
    return [(r, c) for r in range(3) for c in range(3) if board[r][c] == 0]


def find_best_move(board):
    empty_cells = get_empty_cells(board)
    return random.choice(empty_cells) if empty_cells else None


def process_move(board, x, y):
    row, col = int(y // CELL_SIZE), int(x // CELL_SIZE)
    if 0 <= row < 3 and 0 <= col < 3 and board[row][col] == 0:
        return row, col
    return None


@app.route("/process_frame", methods=["POST"])
def process_frame():
    if "board" not in session:
        init_game()

    board = session["board"]
    game_over = session["game_over"]

    file = request.files["frame"]
    frame = np.frombuffer(file.read(), dtype=np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    move_made = False

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        index_tip = hand_landmarks.landmark[8]

        x, y = int(index_tip.x * (CELL_SIZE * 3)), int(index_tip.y * (CELL_SIZE * 3))
        move = process_move(board, x, y)

        if move and not game_over:
            row, col = move
            board[row][col] = HUMAN_PLAYER
            move_made = True

            winner = check_winner(board)
            if winner is not None:
                session["winner"] = winner
                session["game_over"] = True
            else:
                ai_move = find_best_move(board)
                if ai_move:
                    board[ai_move[0]][ai_move[1]] = COMPUTER_PLAYER

                winner = check_winner(board)
                if winner is not None:
                    session["winner"] = winner
                    session["game_over"] = True

    session["board"] = board
    return jsonify(
        {
            "board": board,
            "game_over": session["game_over"],
            "winner": session["winner"],
            "move_made": move_made,
        }
    )


@app.route("/get_game_state", methods=["GET"])
def get_game_state():
    if "board" not in session:
        init_game()
    return jsonify(
        {
            "board": session["board"],
            "game_over": session["game_over"],
            "winner": session["winner"],
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
