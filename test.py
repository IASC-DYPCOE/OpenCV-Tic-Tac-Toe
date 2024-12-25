import cv2
import numpy as np
import mediapipe as mp
import random
from enum import Enum


class GameState(Enum):
    PLAYING = 1
    WON = 2
    LOST = 3


class WumpusWorld:
    def __init__(self):
        self.grid_size = 6
        self.cell_size = 100
        self.window_size = self.grid_size * self.cell_size

        self.board = np.zeros((self.grid_size, self.grid_size))
        self.player_pos = [self.grid_size - 1, 0]
        self.player_direction = 0
        self.wumpus_pos = None
        self.gold_pos = None
        self.pits = []
        self.game_state = GameState.PLAYING

        self.mp_hands = mp.solutions.hands
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(min_detection_confidence=0.7)
        self.previous_x = 0
        self.previous_y = 0

        self.initialize_game()

    def reset_game(self):
        self.player_pos = [self.grid_size - 1, 0]
        self.player_direction = 0
        self.game_state = GameState.PLAYING
        self.initialize_game()

    def initialize_game(self):
        self.wumpus_pos = [
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        ]
        while self.wumpus_pos == [self.grid_size - 1, 0]:
            self.wumpus_pos = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]

        self.gold_pos = [
            random.randint(0, self.grid_size - 1),
            random.randint(0, self.grid_size - 1),
        ]
        while (
            self.gold_pos == [self.grid_size - 1, 0] or self.gold_pos == self.wumpus_pos
        ):
            self.gold_pos = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]

        self.pits = []
        for _ in range(3):
            pit_pos = [
                random.randint(0, self.grid_size - 1),
                random.randint(0, self.grid_size - 1),
            ]
            while (
                pit_pos == [self.grid_size - 1, 0]
                or pit_pos == self.wumpus_pos
                or pit_pos == self.gold_pos
                or pit_pos in self.pits
            ):
                pit_pos = [
                    random.randint(0, self.grid_size - 1),
                    random.randint(0, self.grid_size - 1),
                ]
            self.pits.append(pit_pos)

    def detect_hand_gesture(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style(),
                )

                x = hand_landmarks.landmark[9].x
                y = hand_landmarks.landmark[9].y

                h, w, c = frame.shape
                cx, cy = int(x * w), int(y * h)
                cv2.circle(frame, (cx, cy), 10, (255, 0, 0), -1)
                cv2.putText(
                    frame,
                    "Palm Center",
                    (cx + 10, cy),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 0, 0),
                    1,
                )

                dx = x - self.previous_x
                dy = y - self.previous_y

                self.previous_x = x
                self.previous_y = y

                threshold = 0.02
                if abs(dx) > abs(dy):
                    if dx > threshold:
                        self.move_player("right")
                    elif dx < -threshold:
                        self.move_player("left")
                else:
                    if dy > threshold:
                        self.move_player("down")
                    elif dy < -threshold:
                        self.move_player("up")

    def move_player(self, direction):
        new_pos = self.player_pos.copy()

        if direction == "up":
            self.player_direction = 0
            new_pos[0] -= 1
        elif direction == "right":
            self.player_direction = 90
            new_pos[1] += 1
        elif direction == "down":
            self.player_direction = 180
            new_pos[0] += 1
        elif direction == "left":
            self.player_direction = 270
            new_pos[1] -= 1

        if 0 <= new_pos[0] < self.grid_size and 0 <= new_pos[1] < self.grid_size:
            self.player_pos = new_pos
            self.check_game_state()

    def check_game_state(self):
        if self.player_pos == self.gold_pos:
            self.game_state = GameState.WON
        elif self.player_pos == self.wumpus_pos:
            self.game_state = GameState.LOST
        elif self.player_pos in self.pits:
            self.game_state = GameState.LOST

    def draw_triangle_player(self, board, center, direction):
        size = 20
        angle = np.radians(direction)

        pt1 = (
            int(center[0] + size * np.sin(angle)),
            int(center[1] - size * np.cos(angle)),
        )
        pt2 = (
            int(center[0] + size * np.sin(angle + 2.618)),
            int(center[1] - size * np.cos(angle + 2.618)),
        )
        pt3 = (
            int(center[0] + size * np.sin(angle - 2.618)),
            int(center[1] - size * np.cos(angle - 2.618)),
        )

        pts = np.array([pt1, pt2, pt3], np.int32)
        cv2.fillPoly(board, [pts], (0, 255, 0))

    def draw_game(self):
        board = np.ones((self.window_size, self.window_size, 3), dtype=np.uint8) * 255

        for i in range(self.grid_size):
            cv2.line(
                board,
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size),
                (0, 0, 0),
                2,
            )
            cv2.line(
                board,
                (0, i * self.cell_size),
                (self.window_size, i * self.cell_size),
                (0, 0, 0),
                2,
            )

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                center = (
                    j * self.cell_size + self.cell_size // 2,
                    i * self.cell_size + self.cell_size // 2,
                )

                if [i, j] == self.wumpus_pos:
                    cv2.circle(board, center, 30, (0, 0, 255), -1)

                if [i, j] == self.gold_pos:
                    cv2.circle(board, center, 30, (0, 255, 255), -1)

                if [i, j] in self.pits:
                    cv2.circle(board, center, 30, (128, 128, 128), -1)

                if [i, j] == self.player_pos:
                    self.draw_triangle_player(board, center, self.player_direction)

        cv2.putText(
            board, "Controls:", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )
        cv2.putText(
            board,
            "Hand gestures to move",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            board,
            "Press 'r' to reset",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )
        cv2.putText(
            board,
            "Press 'q' to quit",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
        )

        return board


def main():
    game = WumpusWorld()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        if game.game_state == GameState.PLAYING:
            game.detect_hand_gesture(frame)

        game_board = game.draw_game()

        if game.game_state == GameState.WON:
            cv2.putText(
                game_board,
                "You Won!",
                (game.window_size // 3, game.window_size // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 255, 0),
                3,
            )
        elif game.game_state == GameState.LOST:
            cv2.putText(
                game_board,
                "Game Over!",
                (game.window_size // 3, game.window_size // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.5,
                (0, 0, 255),
                3,
            )

        cv2.imshow("Hand Controls (Press 'q' to quit)", frame)
        cv2.imshow("Wumpus World", game_board)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            game.reset_game()

    cap.release()
    cv2.destroyAllWindows()
    game.hands.close()


if __name__ == "__main__":
    main()
