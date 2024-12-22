# Tic Tac Toe with Hand Tracking 🎮🤖🎉

This project implements a **Tic Tac Toe** game with hand tracking using **OpenCV**, **MediaPipe**, and **NumPy**. The player interacts with the game board via hand gestures detected by a webcam. ✋✨📹

---

## Features 🕹️🛠️🎨
- **Hand Tracking**: Uses MediaPipe to detect hand landmarks and recognize gestures.
- **Pinch Detection**: Detects pinch gestures to make a move on the board.
- **AI Opponent**: Implements a computer player using the Minimax algorithm for optimal gameplay.
- **Interactive GUI**: Displays the game board and live webcam feed in real-time.
- **Game States**: Supports win, lose, and draw conditions with appropriate messages.

---

## Requirements 📦💻🛠️

### Python Libraries
Install the following libraries before running the game:
- **OpenCV**
- **NumPy**
- **MediaPipe**

Install them using the command:
```bash
pip install opencv-python mediapipe numpy
```

### Hardware
- A computer with a webcam. 📷💻🎯

---

## How It Works 🧠🖐️🎲

1. **Board Representation**: 🗂️🔢🟩
   - The board is a 3x3 matrix initialized with zeros.
   - The human player’s moves are represented by `1`, and the computer’s moves by `-1`.

2. **Hand Tracking**: 🤲📍📊
   - MediaPipe detects hand landmarks in the webcam feed.
   - The tip of the index finger is tracked to interact with the game board.

3. **Pinch Detection**: 🌀🖇️✔️
   - A "pinch" gesture is detected when the thumb and index finger come close together.
   - This triggers a move on the game board if the hand is over a valid cell.

4. **AI Opponent**: 🧩🤖♟️
   - The computer uses the Minimax algorithm to decide its moves for optimal gameplay.

5. **Game Logic**: 🎮💡✅
   - The game checks for a winner after each move.
   - The game ends with a win, loss, or draw, and displays the result.

6. **GUI**: 🖼️📊🎨
   - The game board is drawn using OpenCV, showing X (cross) for the human player and O (circle) for the computer.
   - The webcam feed is shown alongside the game board.

---

## How to Run 🚀📂⚙️

1. Clone or download this repository. 📥📂✅
2. Ensure all dependencies are installed.
3. Run the script using:
   ```bash
   python tictactoe.py
   ```
4. The game will open two windows:
   - **Hand Tracking**: Displays the live webcam feed with hand landmarks.
   - **Tic Tac Toe!!**: Displays the game board.

5. Use your hand to interact with the game: ✋✨🤩
   - Hover your index finger over a cell.
   - Pinch your thumb and index finger to place an X in the selected cell.

6. The computer will respond with its move (O). 🤖⭕✅
7. Press `q` to quit the game.

---

## Key Functions 🔑📜🎲

### Game Logic 🧠🎯✔️
- **`check_winner`**: Checks if there is a winner or a draw.
- **`get_empty_cells`**: Returns a list of empty cells on the board.
- **`is_terminal_state`**: Determines if the game has reached a terminal state (win, lose, or draw).
- **`minimax`**: Implements the Minimax algorithm for AI decision-making.

### Hand Tracking ✋📍🌀
- **`is_pinching`**: Calculates the distance between thumb and index fingertips to detect a pinch gesture.
- **`get_cell_from_coordinates`**: Maps hand coordinates to a specific cell on the board.

### Rendering 🎨🖼️✅
- **`draw_board`**: Draws the Tic Tac Toe grid and updates it with the player’s and computer’s moves.

---

## Customization 🔧⚙️✨

### Difficulty Adjustment 🤖🛠️🎯
To modify the difficulty of the computer player, adjust the depth of the `minimax` function or implement a heuristic evaluation.

### Board Size 🗂️🟩📐
To change the board size or cell dimensions, modify the `cell_size` and `board_size` variables in the `__init__` method.

---

## Known Issues 🐛⚠️💡
- **Lighting Conditions**: Hand tracking accuracy may vary based on lighting.
- **Gesture Sensitivity**: Adjusting the pinch detection threshold (`0.05`) may improve usability.

---

## Future Improvements 🌟📈🚀
- Multi-hand support for two-player mode.
- Enhanced gesture controls for a more immersive experience.
- Adding animations or sound effects.
- Optimizing AI for larger boards.

---

## Acknowledgments 🤝🎉📚
- MediaPipe for hand tracking.
- OpenCV for image processing and GUI rendering. 🎨✨🔍

