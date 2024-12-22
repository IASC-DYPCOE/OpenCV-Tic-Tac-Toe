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
        self.game_board = np.zeros(
            (self.board_size, self.board_size, 3), dtype=np.uint8
        )

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

                if self.board[i, j] == 1:
                    cv2.line(
                        self.game_board,
                        (center[0] - 60, center[1] - 60),
                        (center[0] + 60, center[1] + 60),
                        (255, 0, 0),
                        3,
                    )
                    cv2.line(
                        self.game_board,
                        (center[0] + 60, center[1] - 60),
                        (center[0] - 60, center[1] + 60),
                        (255, 0, 0),
                        3,
                    )
                elif self.board[i, j] == -1:
                    cv2.circle(self.game_board, center, 60, (0, 0, 255), 3)

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

    def get_empty_cells(self, board):
        return list(zip(*np.where(board == 0)))

    def is_terminal_state(self, board):
        winner = self.check_winner()
        if winner is not None:
            return True
        return len(self.get_empty_cells(board)) == 0

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
            for row, col in self.get_empty_cells(board):
                board[row, col] = self.computer_player
                score = self.minimax(board, depth + 1, False)
                board[row, col] = 0
                best_score = max(score, best_score)
            return best_score
        else:
            best_score = float("inf")
            for row, col in self.get_empty_cells(board):
                board[row, col] = self.human_player
                score = self.minimax(board, depth + 1, True)
                board[row, col] = 0
                best_score = min(score, best_score)
            return best_score
