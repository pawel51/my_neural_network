import numpy as np
from matplotlib import pyplot as plt


# average difference between all Loss functions
# after single minibatch, from number of iterations
# def loss_plot(train_loss, valid_loss):
#
#     fig = go.Figure()
#     fig.add_trace(go.Scatter(x=np.arange(len(train_loss)),
#                              y=train_loss,
#                              mode='lines',
#                              name='training loss'))
#     fig.add_trace(go.Scatter(x=np.arange(len(valid_loss)),
#                              y=valid_loss,
#                              mode='lines',
#                              name='validation loss'))
#
#     fig.show()


def loss_plot(train_loss, valid_loss):
    x = np.arange(len(train_loss))
    plt.plot(x, train_loss, x, valid_loss)
    plt.show()



# correct outputs divided by all outputs in single minibatch,
# from number of iterations
def accuracy_plot():
    pass
