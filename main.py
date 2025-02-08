import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import random
import math
from PIL import Image
import time


def init_session_state():
    if "board" not in st.session_state:
        st.session_state.board = np.zeros((3, 3), dtype=int)
    if "game_over" not in st.session_state:
        st.session_state.game_over = False
    if "winner" not in st.session_state:
        st.session_state.winner = None
    if "difficulty" not in st.session_state:
        st.session_state.difficulty = "medium"
    if "last_move_time" not in st.session_state:
        st.session_state.last_move_time = time.time()
    if "cell_size" not in st.session_state:
        st.session_state.cell_size = 200
    if "human_player" not in st.session_state:
        st.session_state.human_player = 1
    if "computer_player" not in st.session_state:
        st.session_state.computer_player = -1
    if "show_results" not in st.session_state:
        st.session_state.show_results = False


def reset_game():
    st.session_state.board = np.zeros((3, 3), dtype=int)
    st.session_state.game_over = False
    st.session_state.winner = None
    st.session_state.last_move_time = time.time()
    st.session_state.show_results = False


def get_empty_cells():
    return list(zip(*np.where(st.session_state.board == 0)))


def is_pinching(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    distance = math.sqrt(
        (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
    )
    return distance < 0.05


def get_cell_from_coordinates(x, y):
    board_size = st.session_state.cell_size * 3
    row = int(y * 3 // board_size)
    col = int(x * 3 // board_size)
    return row, col


def check_winner():
    board = st.session_state.board

    # Check rows and columns
    for i in range(3):
        if abs(sum(board[i, :])) == 3:
            return board[i, 0]
        if abs(sum(board[:, i])) == 3:
            return board[0, i]

    # Check diagonals
    if abs(sum(np.diag(board))) == 3:
        return board[0, 0]
    if abs(sum(np.diag(np.fliplr(board)))) == 3:
        return board[0, 2]

    # Check for draw
    if np.count_nonzero(board) == 9:
        return 0

    return None


def find_best_move():
    if st.session_state.difficulty == "easy":
        return random.choice(get_empty_cells())
    elif st.session_state.difficulty == "medium":
        return find_medium_move()
    else:
        return find_hard_move()


def find_medium_move():
    # Check for winning move
    for row, col in get_empty_cells():
        st.session_state.board[row, col] = st.session_state.computer_player
        if check_winner() == st.session_state.computer_player:
            st.session_state.board[row, col] = 0
            return (row, col)
        st.session_state.board[row, col] = 0

    # Check for blocking move
    for row, col in get_empty_cells():
        st.session_state.board[row, col] = st.session_state.human_player
        if check_winner() == st.session_state.human_player:
            st.session_state.board[row, col] = 0
            return (row, col)
        st.session_state.board[row, col] = 0

    return random.choice(get_empty_cells())


def minimax(board, depth, is_maximizing):
    winner = check_winner()
    if winner == st.session_state.computer_player:
        return 1
    elif winner == st.session_state.human_player:
        return -1
    elif winner == 0:
        return 0

    if is_maximizing:
        best_score = float("-inf")
        for row, col in get_empty_cells():
            board[row, col] = st.session_state.computer_player
            score = minimax(board, depth + 1, False)
            board[row, col] = 0
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float("inf")
        for row, col in get_empty_cells():
            board[row, col] = st.session_state.human_player
            score = minimax(board, depth + 1, True)
            board[row, col] = 0
            best_score = min(score, best_score)
        return best_score


def find_hard_move():
    best_score = float("-inf")
    best_move = None
    for row, col in get_empty_cells():
        st.session_state.board[row, col] = st.session_state.computer_player
        score = minimax(st.session_state.board, 0, False)
        st.session_state.board[row, col] = 0
        if score > best_score:
            best_score = score
            best_move = (row, col)
    return best_move


def draw_board():
    board_size = st.session_state.cell_size * 3
    game_board = np.full((board_size, board_size, 3), 255, dtype=np.uint8)

    # Draw grid lines
    for i in range(1, 3):
        cv2.line(
            game_board,
            (i * st.session_state.cell_size, 0),
            (i * st.session_state.cell_size, board_size),
            (0, 0, 0),
            2,
        )
        cv2.line(
            game_board,
            (0, i * st.session_state.cell_size),
            (board_size, i * st.session_state.cell_size),
            (0, 0, 0),
            2,
        )

    # Draw X's and O's
    for i in range(3):
        for j in range(3):
            center = (
                j * st.session_state.cell_size + st.session_state.cell_size // 2,
                i * st.session_state.cell_size + st.session_state.cell_size // 2,
            )

            if st.session_state.board[i, j] == st.session_state.human_player:  # X
                cv2.line(
                    game_board,
                    (center[0] - 60, center[1] - 60),
                    (center[0] + 60, center[1] + 60),
                    (0, 0, 255),
                    3,
                )
                cv2.line(
                    game_board,
                    (center[0] + 60, center[1] - 60),
                    (center[0] - 60, center[1] + 60),
                    (0, 0, 255),
                    3,
                )
            elif st.session_state.board[i, j] == st.session_state.computer_player:  # O
                cv2.circle(game_board, center, 60, (255, 0, 0), 3)

    return game_board


def main():
    st.set_page_config(layout="wide", page_title="Hand-Controlled Tic Tac Toe")

    # Initialize session state
    init_session_state()

    # Sidebar controls
    with st.sidebar:
        st.header("Game Controls")

        # Difficulty selector
        difficulty = st.selectbox(
            "Select Difficulty",
            ["easy", "medium", "hard"],
            index=["easy", "medium", "hard"].index(st.session_state.difficulty),
        )
        st.session_state.difficulty = difficulty

        # Reset button
        if st.button("Reset Game", type="primary"):
            reset_game()

        # Instructions
        st.markdown("---")
        st.header("How to Play")
        st.markdown("""
        1. Allow camera access when prompted
        2. Use your hand to control the game:
           - Move your index finger to select a cell
           - Pinch your thumb and index finger to make a move
        3. Your moves are marked with X (red)
        4. Computer moves are marked with O (blue)
        """)

        # Quit button
        if st.button("Quit Game", type="secondary"):
            st.session_state.clear()
            st.experimental_rerun()

    # Main game layout
    col1, col2 = st.columns([2, 1])

    # Game board column
    with col1:
        board_placeholder = st.empty()
        status_placeholder = st.empty()

    # Camera feed column
    with col2:
        camera_placeholder = st.empty()
        camera_status = st.empty()

    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    mp_draw = mp.solutions.drawing_utils

    # Start video capture
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        camera_status.error(
            "Failed to access webcam. Please check your camera connection."
        )
        return

    # Main game loop
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                camera_status.error("Failed to capture video frame.")
                break

            # Process frame for hand tracking
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            # Draw current game board
            game_board = draw_board()

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get index finger position
                index_tip = hand_landmarks.landmark[8]
                board_size = st.session_state.cell_size * 3
                x = int(index_tip.x * board_size)
                y = int(index_tip.y * board_size)

                # Show cursor position on game board
                cv2.circle(game_board, (x, y), 10, (0, 255, 0), -1)

                # Check for pinch gesture
                current_time = time.time()
                if (
                    is_pinching(hand_landmarks)
                    and current_time - st.session_state.last_move_time > 1.0
                    and not st.session_state.game_over
                ):
                    row, col = get_cell_from_coordinates(x, y)
                    if (
                        0 <= row < 3
                        and 0 <= col < 3
                        and st.session_state.board[row, col] == 0
                    ):
                        # Make human move
                        st.session_state.board[row, col] = st.session_state.human_player
                        st.session_state.last_move_time = current_time

                        # Check for winner after human move
                        winner = check_winner()
                        if winner is None:
                            # Make computer move
                            computer_move = find_best_move()
                            if computer_move:
                                st.session_state.board[
                                    computer_move[0], computer_move[1]
                                ] = st.session_state.computer_player

            # Check for game over
            winner = check_winner()
            if winner is not None and not st.session_state.game_over:
                st.session_state.game_over = True
                st.session_state.winner = winner

            # Display game status with enhanced UI
            if st.session_state.game_over and not st.session_state.show_results:
                st.session_state.show_results = True
                if st.session_state.winner == 0:
                    with status_placeholder.container():
                        st.title("Game Over - It's a Draw!")
                        if st.button("Play Again", key="draw_restart"):
                            reset_game()
                else:
                    winner_text = (
                        "Congratulations! You Win! 🎉"
                        if st.session_state.winner == st.session_state.human_player
                        else "Computer Wins! Try Again?"
                    )
                    if st.session_state.winner == st.session_state.human_player:
                        with status_placeholder.container():
                            st.title(winner_text)
                            if st.button("Play Again", key="win_restart"):
                                reset_game()
                    else:
                        with status_placeholder.container():
                            st.title(winner_text)
                            if st.button("Play Again", key="lose_restart"):
                                reset_game()

            # Update displays
            camera_placeholder.image(frame, channels="RGB", use_container_width=True)
            board_placeholder.image(
                game_board, channels="RGB", use_container_width=True
            )

            # Add small delay to prevent excessive CPU usage
            time.sleep(0.01)

    finally:
        # Properly release resources
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
