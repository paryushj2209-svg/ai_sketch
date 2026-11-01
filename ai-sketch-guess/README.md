# AI Sketch Guess
A small VGG^ CNN that guesses doodles drawn on a 512×512 canvas.
The classifier uses Google's Quick, Draw! dataset.
Install dependencies: `pip install -r requirement.txt`
Download classes: `python src/guess/data.py --classes cat dog`
Train the model: `python src/guess/guess.py train --classes cat dog --download`
Start the canvas: `python src/test/canvas.py`
Predictions are printed in the terminal every 1.5 seconds.
The game offers Hexagon only. Each level has a 30-second timer; when time ends, the drawing locks and its model score appears with a next-level option for Levels 1 and 2.
Level 1 uses the mouse, Level 2 uses the webcam (pinch to draw), and Level 3 uses WASD or arrow keys (hold Space to draw). Level 3 wins at 90% or above; avoid 15–25 invisible bombs.
The interface uses a dark neural-network theme; the camera preview stays hidden, the Level 3 car turns as it moves, and bomb hits trigger an animated explosion.
