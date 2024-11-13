import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

# Function to create the board containing the information of the selected features
def create_board(n_rows, n_cols, final_sequence, sequence_length, feat_sel):
    board = np.zeros((n_rows, n_cols))

    for i in range(n_rows):
        start_index = int(final_sequence[i]) 
        end_index = int(final_sequence[i]) + int(sequence_length[i])
        if feat_sel[i] != 0:
            board[i, start_index:end_index] = 1
    
    return board

# Function to plot the board with the selected features, at which time lags and to higlight the non-selected features
def plot_board(board, column_names, feat_sel):
    fig, ax = plt.subplots(figsize=(5, 14))
    ax.imshow(np.flip(board, axis=0), cmap='Blues', origin='lower', aspect='auto')
    
    ax.xaxis.set_label_position("top") 
    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(board.shape[1]))
    ax.set_xticklabels(np.arange(board.shape[1]), fontsize=11)

    ax.set_yticks(np.arange(len(column_names)))
    ax.set_yticklabels(np.flip(np.asarray(column_names)), fontsize=11)

    minor_locator = AutoMinorLocator(2)
    ax.yaxis.set_minor_locator(minor_locator)
    ax.yaxis.grid(which='minor',color='black', linewidth=1)
    ax.xaxis.grid(which='minor',color='black', linewidth=1)
    ax.set_xlabel('Time lags (months)', fontsize=15)

    for i in range(board.shape[0]):
        pos = board.shape[0] - i - 1.5
        if feat_sel[i] == 0:
            rect = plt.Rectangle((- 0.5, pos), 1, 1, color='red')
            ax.add_patch(rect)

    plt.tight_layout()
    return fig