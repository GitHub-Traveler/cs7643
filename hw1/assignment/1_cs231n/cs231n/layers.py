import numpy as np

def affine_forward(x, w, b):
  """
  Computes the forward pass for an affine (fully-connected) layer.

  The input x has shape (N, d_1, ..., d_k) where x[i] is the ith input.
  We multiply this against a weight matrix of shape (D, M) where
  D = \prod_i d_i

  Inputs:
  x - Input data, of shape (N, d_1, ..., d_k)
  w - Weights, of shape (D, M)
  b - Biases, of shape (M,)
  
  Returns a tuple of:
  - out: output, of shape (N, M)
  - cache: (x, w, b)
  """
  out = None
  #############################################################################
  # TODO: Implement the affine forward pass. Store the result in out. You     #
  # will need to reshape the input into rows.                                 #
  #############################################################################
  out = np.matmul(x.reshape(-1, w.shape[0]), w) + b
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b)
  return out, cache


def affine_backward(dout, cache):
  """
  Computes the backward pass for an affine layer.

  Inputs:
  - dout: Upstream derivative, of shape (N, M)
  - cache: Tuple of:
    - x: Input data, of shape (N, d_1, ... d_k)
    - w: Weights, of shape (D, M)

  Returns a tuple of:
  - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
  - dw: Gradient with respect to w, of shape (D, M)
  - db: Gradient with respect to b, of shape (M,)
  """
  x, w, b = cache
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the affine backward pass.                                 #
  #############################################################################
  dx = np.reshape(np.matmul(dout, w.T), x.shape)
  dw = np.matmul(x.reshape(-1, w.shape[0]).T, dout)
  db = np.sum(dout, axis = 0)
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def relu_forward(x):
  """
  Computes the forward pass for a layer of rectified linear units (ReLUs).

  Input:
  - x: Inputs, of any shape

  Returns a tuple of:
  - out: Output, of the same shape as x
  - cache: x
  """
  out = None
  #############################################################################
  # TODO: Implement the ReLU forward pass.                                    #
  #############################################################################
  out = np.maximum(0, x)
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = x
  return out, cache


def relu_backward(dout, cache):
  """
  Computes the backward pass for a layer of rectified linear units (ReLUs).

  Input:
  - dout: Upstream derivatives, of any shape
  - cache: Input x, of same shape as dout

  Returns:
  - dx: Gradient with respect to x
  """
  dx, x = None, cache
  #############################################################################
  # TODO: Implement the ReLU backward pass.                                   #
  #############################################################################
  dx = dout
  dx[x <= 0] = 0
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def conv_forward_naive(x, w, b, conv_param):
  """
  A naive implementation of the forward pass for a convolutional layer.

  The input consists of N data points, each with C channels, height H and width
  W. We convolve each input with F different filters, where each filter spans
  all C channels and has height HH and width HH.

  Input:
  - x: Input data of shape (N, C, H, W)
  - w: Filter weights of shape (F, C, HH, WW)
  - b: Biases, of shape (F,)
  - conv_param: A dictionary with the following keys:
    - 'stride': The number of pixels between adjacent receptive fields in the
      horizontal and vertical directions.
    - 'pad': The number of pixels that will be used to zero-pad the input.

  Returns a tuple of:
  - out: Output data, of shape (N, F, H', W') where H' and W' are given by
    H' = 1 + (H + 2 * pad - HH) / stride
    W' = 1 + (W + 2 * pad - WW) / stride
  - cache: (x, w, b, conv_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the convolutional forward pass.                           #
  # Hint: you can use the function np.pad for padding.                        #
  #############################################################################
  F, _, HH, WW = w.shape
  # new_H = 1 + (H + 2 * conv_param['pad'] - HH) / conv_param['stride']
  # new_W = 1 + (W + 2 * conv_param['pad'] - WW) / conv_param['stride']
  padding, stride = conv_param['pad'], conv_param['stride']
  padded_image = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
  N, C, H, W = padded_image.shape
  out_H = (H - HH) // stride + 1
  out_W = (W - WW) // stride + 1
  col = np.zeros((out_H * out_W * N, HH * WW * C))
  for i in range(N):
    for j in range(out_H):
      for k in range(out_W):
        col[i * out_W * out_H + j * out_W + k] = padded_image[i, :, j*stride:j*stride+HH, k*stride:k*stride+WW].flatten()
  kernel_col = np.reshape(w, (F, -1))
  kernel_col = kernel_col.T
  out = np.matmul(col, kernel_col) + b
  out = np.transpose(np.reshape(out, (N, out_H, out_W, F)), (0, 3, 1, 2))
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, w, b, conv_param)
  return out, cache


def conv_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a convolutional layer.

  Inputs:
  - dout: Upstream derivatives.
  - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

  Returns a tuple of:
  - dx: Gradient with respect to x
  - dw: Gradient with respect to w
  - db: Gradient with respect to b
  """
  dx, dw, db = None, None, None
  #############################################################################
  # TODO: Implement the convolutional backward pass.                          #
  #############################################################################
  x, w, b, conv_param = cache
  F, _, HH, WW = w.shape
  padding, stride = conv_param['pad'], conv_param['stride']
  padded_image = np.pad(x, ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
  N, C, H, W = padded_image.shape
  out_H = (H - HH) // stride + 1
  out_W = (W - WW) // stride + 1
  col = np.zeros((out_H * out_W * N, HH * WW * C))
  for i in range(N):
    for j in range(out_H):
      for k in range(out_W):
        col[i * out_W * out_H + j * out_W + k] = padded_image[i, :, j*stride:j*stride+HH, k*stride:k*stride+WW].flatten()
  kernel_col = np.reshape(w, (F, -1))
  kernel_col = kernel_col.T
  
  out_channel = dout.shape[1]
  dout_matrix = dout.transpose(0, 2, 3, 1).reshape(-1, out_channel)
  db = dout_matrix.sum(axis = 0)
  dw = np.matmul(col.T, dout_matrix).reshape(C, HH, WW, F).transpose(3, 0, 1, 2)
  dx_matrix = np.matmul(dout_matrix, kernel_col.T)
  
  dx = np.pad(np.zeros((N, C, H, W)), ((0, 0), (0, 0), (padding, padding), (padding, padding)), mode='constant')
  for i in range(N):
    for j in range(out_H):
      for k in range(out_W):
        dx[i, :, j*stride:j*stride+HH, k*stride:k*stride+WW] += dx_matrix[i * out_W * out_H + j * out_W + k].reshape(C, HH, WW)
  dx = dx[:, :, padding:H-padding, padding:W-padding]
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx, dw, db


def max_pool_forward_naive(x, pool_param):
  """
  A naive implementation of the forward pass for a max pooling layer.

  Inputs:
  - x: Input data, of shape (N, C, H, W)
  - pool_param: dictionary with the following keys:
    - 'pool_height': The height of each pooling region
    - 'pool_width': The width of each pooling region
    - 'stride': The distance between adjacent pooling regions

  Returns a tuple of:
  - out: Output data
  - cache: (x, pool_param)
  """
  out = None
  #############################################################################
  # TODO: Implement the max pooling forward pass                              #
  #############################################################################
  N, C, H, W = x.shape
  HH, WW, stride = pool_param.values()
  new_H = (H - HH) // stride + 1
  new_W = (W - WW) // stride + 1
  out = np.zeros((N, C, new_H, new_W))
  for i in range(N):
    for j in range(C):
      for k in range(new_H):
        for l in range(new_W):
          out[i,j,k,l] = np.max(x[i,j,k*stride:k*stride+HH, l*stride:l*stride+WW])
  # pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  cache = (x, pool_param)
  return out, cache


def max_pool_backward_naive(dout, cache):
  """
  A naive implementation of the backward pass for a max pooling layer.

  Inputs:
  - dout: Upstream derivatives
  - cache: A tuple of (x, pool_param) as in the forward pass.

  Returns:
  - dx: Gradient with respect to x
  """
  dx = None
  #############################################################################
  # TODO: Implement the max pooling backward pass                             #
  #############################################################################
  x, pool_param = cache
  HH, WW, stride = pool_param.values()
  N, C, new_H, new_W = dout.shape
  N, C, H, W = x.shape
  dx = np.zeros((N, C, H, W))
  for i in range(N):
    for j in range(C):
      for k in range(new_H):
        for l in range(new_W):
          index = np.argmax(x[i,j,k*stride:k*stride+HH, l*stride:l*stride+WW])
          index_h, index_w = divmod(index, WW)
          dx[i,j,k*stride+index_h,l*stride+index_w] = dout[i, j, k, l]
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  return dx


def svm_loss(x, y):
  """
  Computes the loss and gradient using for multiclass SVM classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  N = x.shape[0]
  correct_class_scores = x[np.arange(N), y]
  margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
  margins[np.arange(N), y] = 0
  loss = np.sum(margins) / N
  num_pos = np.sum(margins > 0, axis=1)
  dx = np.zeros_like(x)
  dx[margins > 0] = 1
  dx[np.arange(N), y] -= num_pos
  dx /= N
  return loss, dx


def softmax_loss(x, y):
  """
  Computes the loss and gradient for softmax classification.

  Inputs:
  - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
    for the ith input.
  - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
    0 <= y[i] < C

  Returns a tuple of:
  - loss: Scalar giving the loss
  - dx: Gradient of the loss with respect to x
  """
  probs = np.exp(x - np.max(x, axis=1, keepdims=True))
  probs /= np.sum(probs, axis=1, keepdims=True)
  N = x.shape[0]
  loss = -np.sum(np.log(probs[np.arange(N), y])) / N
  dx = probs.copy()
  dx[np.arange(N), y] -= 1
  dx /= N
  return loss, dx

