import cv2
import numpy as np
import mediapipe as mp
import random
import math


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.human_player = 1
        self.computer_player = -1
        self.game_over = False
        self.winner = None
        self.difficulty = "medium"

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
        )
        self.mp_draw = mp.solutions.drawing_utils

        self.cell_size = 200
        self.board_size = self.cell_size * 3
        self.game_board = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)

    def set_difficulty(self):
        difficulty_window = np.zeros((200, 400, 3), dtype=np.uint8)  # Create a blank window
        cv2.putText(difficulty_window, "Select Difficulty:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(difficulty_window, "E - Easy", (50, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(difficulty_window, "M - Medium", (50, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(difficulty_window, "H - Hard", (50, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Difficulty Selection", difficulty_window)

        while True:
            key = cv2.waitKey(0) & 0xFF  # Wait indefinitely for key press
            if key == ord("e"):
                self.difficulty = "easy"
                break
            elif key == ord("m"):
                self.difficulty = "medium"
                break
            elif key == ord("h"):
                self.difficulty = "hard"
                break

        cv2.destroyWindow("Difficulty Selection")  # Close the selection window
        print(f"Difficulty set to: {self.difficulty}")

    def reset_game(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.game_over = False
        self.winner = None
        self.game_board = np.zeros((self.board_size, self.board_size, 3), dtype=np.uint8)

    def get_empty_cells(self):
        return list(zip(*np.where(self.board == 0)))
    
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

        for i in range(3):
            for j in range(3):
                center = (
                    j * self.cell_size + self.cell_size // 2,
                    i * self.cell_size + self.cell_size // 2,
                )

                if self.board[i, j] == 1:  # Human (X)
                    cv2.line(
                        self.game_board,
                        (center[0] - 60, center[1] - 60),
                        (center[0] + 60, center[1] + 60),
                        (0, 0, 255),  # Changed color to red for human
                        3,
                    )
                    cv2.line(
                        self.game_board,
                        (center[0] + 60, center[1] - 60),
                        (center[0] - 60, center[1] + 60),
                        (0, 0, 255),  # Changed color to red for human
                        3,
                    )
                elif self.board[i, j] == -1:  # Computer (O)
                    cv2.circle(
                        self.game_board, center, 60, (255, 0, 0), 3
                    )  # Changed color to blue for computer


    def minimax(self, board, depth, is_maximizing):
        winner = self.check_winner()
        if winner == self.computer_player:
            return 1
        elif winner == self.human_player:
            return -1
        elif winner == 0:
            return 0

        if is_maximizing:
            best_score = float("-inf")
            for row, col in self.get_empty_cells():
                board[row, col] = self.computer_player
                score = self.minimax(board, depth + 1, False)
                board[row, col] = 0
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float("inf")
            for row, col in self.get_empty_cells():
                board[row, col] = self.human_player
                score = self.minimax(board, depth + 1, True)
                board[row, col] = 0
                best_score = min(score, best_score)
            return best_score

    def computer_move(self):
        if self.difficulty == "easy":
            move = random.choice(self.get_empty_cells())
        elif self.difficulty == "medium":
            move = self.find_best_medium_move()
        else:
            move = self.find_best_hard_move()

        if move:
            self.board[move[0], move[1]] = self.computer_player

    def find_best_medium_move(self):
        for row, col in self.get_empty_cells():
            self.board[row, col] = self.computer_player
            if self.check_winner() == self.computer_player:
                return (row, col)
            self.board[row, col] = 0

        for row, col in self.get_empty_cells():
            self.board[row, col] = self.human_player
            if self.check_winner() == self.human_player:
                return (row, col)
            self.board[row, col] = 0

        return random.choice(self.get_empty_cells())

    def find_best_hard_move(self):
        best_score = float("-inf")
        best_move = None
        for row, col in self.get_empty_cells():
            self.board[row, col] = self.computer_player
            score = self.minimax(self.board, 0, False)
            self.board[row, col] = 0
            if score > best_score:
                best_score = score
                best_move = (row, col)
        return best_move

    def check_winner(self):
        for i in range(3):
            if abs(sum(self.board[i, :])) == 3:
                return self.board[i, 0]
            if abs(sum(self.board[:, i])) == 3:
                return self.board[0, i]

        if abs(sum(np.diag(self.board))) == 3:
            return self.board[0, 0]
        if abs(sum(np.diag(np.fliplr(self.board)))) == 3:
            return self.board[0, 2]

        if np.count_nonzero(self.board) == 9:
            return 0

        return None

    def play(self):
        cap = cv2.VideoCapture(0)
        last_move_time = 0
        cooldown = 1.0

        # Set difficulty before starting the game
        self.set_difficulty()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)

            self.draw_board()

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                index_tip = hand_landmarks.landmark[8]  # Index finger tip
                x = int(index_tip.x * self.board_size)
                y = int(index_tip.y * self.board_size)

                cv2.circle(self.game_board, (x, y), 10, (0, 255, 0), -1)

                current_time = cv2.getTickCount() / cv2.getTickFrequency()
                if self.is_pinching(hand_landmarks) and current_time - last_move_time > cooldown and not self.game_over:
                    row, col = self.get_cell_from_coordinates(x, y)
                    if 0 <= row < 3 and 0 <= col < 3 and self.board[row, col] == 0:
                        self.board[row, col] = self.human_player
                        last_move_time = current_time

                        winner = self.check_winner()
                        if winner is None:
                            self.computer_move()  # Now considers selected difficulty

            winner = self.check_winner()
            if winner is not None and not self.game_over:
                self.game_over = True
                self.winner = winner

            if self.game_over:
                if self.winner == 0:
                    text = "Draw!"
                else:
                    text = "Player (X) wins!" if self.winner == self.human_player else "Computer (O) wins!"
                cv2.putText(self.game_board, text, (self.board_size // 4, self.board_size // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
                cv2.putText(self.game_board, "Press 'r' to reset", (self.board_size // 4, self.board_size // 2 + 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            cv2.imshow("Hand Tracking", frame)
            cv2.imshow("Tic Tac Toe", self.game_board)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.reset_game()
                self.set_difficulty()  # Re-select difficulty after resetting

        cap.release()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    game = TicTacToe()
    game.play()
