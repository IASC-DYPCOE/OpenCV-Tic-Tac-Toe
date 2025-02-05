import cv2
import numpy as np
import mediapipe as mp
import math


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.player1 = 1  # X player
        self.player2 = -1  # O player
        self.current_player = self.player1  # Start with player 1
        self.game_over = False
        self.winner = None

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,  # Set to 2 to track both players' hands
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.cell_size = 200
        self.board_size = self.cell_size * 3
        self.game_board = np.zeros(
            (self.board_size, self.board_size, 3), dtype=np.uint8
        )

    def reset_game(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.game_over = False
        self.winner = None
        self.current_player = self.player1
        self.game_board = np.zeros(
            (self.board_size, self.board_size, 3), dtype=np.uint8
        )

    def is_pinching(self, hand_landmarks):
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]

        distance = math.sqrt(
            (thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2
        )

        return distance < 0.05

    def get_cell_from_coordinates(self, x, y):
        row = int(y * 3 // self.board_size)
        col = int(x * 3 // self.board_size)
        return row, col

    def draw_board(self):
        self.game_board.fill(255)

        # Draw grid lines
        for i in range(1, 3):
            cv2.line(
                self.game_board,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.board_size),
                (0, 0, 0),
                2,
            )
            cv2.line(
                self.game_board,
                (0, i * self.cell_size),
                (self.board_size, i * self.cell_size),
                (0, 0, 0),
                2,
            )

        # Draw X's and O's
        for i in range(3):
            for j in range(3):
                center = (
                    j * self.cell_size + self.cell_size // 2,
                    i * self.cell_size + self.cell_size // 2,
                )

                if self.board[i, j] == self.player1:  # X
                    cv2.line(
                        self.game_board,
                        (center[0] - 60, center[1] - 60),
                        (center[0] + 60, center[1] + 60),
                        (0, 0, 255),  # Red for Player 1
                        3,
                    )
                    cv2.line(
                        self.game_board,
                        (center[0] + 60, center[1] - 60),
                        (center[0] - 60, center[1] + 60),
                        (0, 0, 255),
                        3,
                    )
                elif self.board[i, j] == self.player2:  # O
                    cv2.circle(
                        self.game_board,
                        center,
                        60,
                        (255, 0, 0),  # Blue for Player 2
                        3,
                    )

    def check_winner(self):
        # Check rows and columns
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return self.board[i, 0]
            if abs(sum(self.board[:, i])) == 3:
                return self.board[0, i]

        # Check diagonals
        if abs(sum(np.diag(self.board))) == 3:
            return self.board[0, 0]
        if abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            return self.board[0, 2]

        # Check for draw
        if not any(0 in row for row in self.board):
            return 0

        return None

    def play(self):
        cap = cv2.VideoCapture(0)
        last_move_time = 0
        cooldown = 1.0  # 1 second cooldown between moves

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            self.draw_board()

            # Show current player turn
            player_text = (
                "Player 1 (X)"
                if self.current_player == self.player1
                else "Player 2 (O)"
            )
            cv2.putText(
                frame,
                f"{player_text}'s turn",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
            )

            # Add game instructions
            cv2.putText(
                frame,
                "Pinch to place marker",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            current_time = cv2.getTickCount() / cv2.getTickFrequency()

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(
                        frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                    )

                    index_tip = hand_landmarks.landmark[8]
                    x = int(index_tip.x * self.board_size)
                    y = int(index_tip.y * self.board_size)

                    # Show cursor position for active player
                    if 0 <= x < self.board_size and 0 <= y < self.board_size:
                        cursor_color = (
                            (0, 0, 255)
                            if self.current_player == self.player1
                            else (255, 0, 0)
                        )
                        cv2.circle(self.game_board, (x, y), 10, cursor_color, -1)

                    if (
                        self.is_pinching(hand_landmarks)
                        and current_time - last_move_time > cooldown
                        and not self.game_over
                    ):
                        row, col = self.get_cell_from_coordinates(x, y)
                        if 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0:
                            self.board[row, col] = self.current_player
                            last_move_time = current_time
                            # Switch players after valid move
                            self.current_player = (
                                self.player2
                                if self.current_player == self.player1
                                else self.player1
                            )

            winner = self.check_winner()
            if winner is not None and not self.game_over:
                self.game_over = True
                self.winner = winner

            if self.game_over:
                if self.winner == 0:
                    text = "Draw!"
                else:
                    text = f"Player {'1' if self.winner == self.player1 else '2'} wins!"
                cv2.putText(
                    self.game_board,
                    text,
                    (self.board_size // 4, self.board_size // 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 0, 0),
                    3,
                )
                cv2.putText(
                    self.game_board,
                    "Press 'r' to reset",
                    (self.board_size // 4, self.board_size // 2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 0),
                    2,
                )

            cv2.imshow("Hand Tracking", frame)
            cv2.imshow("Tic Tac Toe", self.game_board)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.reset_game()

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    game = TicTacToe()
    game.play()
