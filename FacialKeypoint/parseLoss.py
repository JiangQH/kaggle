import re
import numpy as np
import matplotlib.pyplot as plt
import sys

def parse_log(log_file):
    with open(log_file, 'r') as f:
        log = f.read()

    train_patern = r"Iteration (?P<iter_num>\d+), loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    train_losses = []
    train_iterations = []
    for r in re.findall(train_patern, log):
        train_iterations.append(int(r[0]))
        train_losses.append(float(r[1]))

    train_iterations = np.array(train_iterations[1:])
    train_losses = np.array(train_losses[1:])


    test_patern = r"Iteration (?P<iter_num>\d+), Testing net \(#0\)\n.* loss = (?P<loss_val>[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?)"
    test_losses = []
    test_iterations = []
    for r in re.findall(test_patern, log):
        test_iterations.append(int(r[0]))
        test_losses.append(float(r[1]))

    test_iterations = np.array(test_iterations[1:])
    test_losses = np.array(test_losses[1:])

    return train_iterations, train_losses, test_iterations, test_losses


def disp_results(file_name):
    plt.style.use('ggplot')
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('train_loss')
    ax2.set_ylabel('val_loss')
    train_iterations, train_loss, test_iterations, test_loss = parse_log(file_name)
    ax1.plot(train_iterations, train_loss)
    ax2.plot(test_iterations, test_loss)
    plt.show()
    

if __name__ == '__main__':
    file = sys.argv[1]
    disp_results(file)










