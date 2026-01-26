Project Structure and Directory Descriptions:
blender: Contains the primary Blender project file and the automation script used to generate and render the synthetic environment.
data_scripts: Includes utility scripts for data preprocessing, such as image rendering, frame cropping for synthetic images, and the 8X8 tiling process.
data_test: The destination folder for the final output images generated after the model’s inference process.
fens: A repository of the FEN (Forsyth-Edwards Notation) strings used to define the board positions for synthetic image generation.
outputs: Stores the results of the training process, including visual samples from each epoch, model checkpoints (weights), and CSV files tracking loss metrics to monitor convergence.
temp_runs: A temporary storage directory for images generated during specific test runs based on a given FEN input.
test: Contains the execution script (final_run) used to run the model in inference mode and evaluate its performance on test data.
train: Includes the core source code used for training the model and optimizing the weights.


Execution Instructions:
To run the model in inference mode, execute the final_run script using the following command:
python final_run --fen "fen_string" --side “black” / “white”
