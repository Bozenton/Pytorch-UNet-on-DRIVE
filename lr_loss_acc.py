from cProfile import label
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

if __name__ == '__main__':
    lr_files = [
        'run-Jun03_21-14-16_featurize-tag-learning_rate.csv', 
        'run-Jun03_21-18-41_featurize-tag-learning_rate.csv', 
        'run-Jun03_21-26-17_featurize-tag-learning_rate.csv'
    ]
    loss_files = [
        'run-Jun03_21-14-16_featurize-tag-loss.csv', 
        'run-Jun03_21-18-41_featurize-tag-loss.csv', 
        'run-Jun03_21-26-17_featurize-tag-loss.csv', 
    ]
    val_scores_files = [
        'run-Jun03_21-14-16_featurize-tag-val_score.csv', 
        'run-Jun03_21-18-41_featurize-tag-val_score.csv', 
        'run-Jun03_21-26-17_featurize-tag-val_score.csv',
    ]
    files_dir = '.'

    mpl.rcParams['font.sans-serif'] = 'Times New Roman'
    fig, axs = plt.subplots(3,1, figsize=(6, 6), dpi=100, layout='constrained')

    for i in range(len(lr_files)):
        lr_path = os.path.join(files_dir, lr_files[i])
        loss_path = os.path.join(files_dir, loss_files[i])
        vs_path = os.path.join(files_dir, val_scores_files[i])

        assert os.path.exists(lr_path)
        assert os.path.exists(loss_path)
        assert os.path.exists(vs_path)

        lr_data = pd.read_csv(lr_path)
        loss_data = pd.read_csv(loss_path)
        vs_data = pd.read_csv(vs_path)

        step_lr = np.array(lr_data['Step'])+1
        lr = np.array(lr_data['Value'])
        step = np.array(loss_data['Step'])+1
        loss = np.array(loss_data['Value'])
        vs = np.array(vs_data['Value'])

        axs[0].plot(step_lr, lr, label=f'{i+1}th train')
        # axs[0].set_xlabel('Epochs')  # Add an x-label to the axes.
        axs[0].set_ylabel('Learning Rate')  # Add a y-label to the axes.
        axs[0].set_title("Learning Rate v.s. Epochs")  # Add a title to the axes.

        axs[1].plot(step, loss, label=f'{i+1}th train')
        # axs[1].set_xlabel('Epochs')  # Add an x-label to the axes.
        axs[1].set_ylabel('Loss')  # Add a y-label to the axes.
        axs[1].set_title("Loss v.s. Epochs")  # Add a title to the axes.

        axs[2].plot(step_lr, vs, label=f'{i+1}th train')
        axs[2].set_xlabel('Epochs')  # Add an x-label to the axes.
        axs[2].set_ylabel('Dice Score')  # Add a y-label to the axes.
        axs[2].set_title("Dice Score v.s. Epochs")  # Add a title to the axes.

    axs[0].legend()
    # axs[1].legend()
    # axs[2].legend()
    plt.savefig('lr_loss_dice.pdf', format='pdf', bbox_inches='tight', transparent=True, dpi=200)
    plt.show()


