import math
import matplotlib
from matplotlib import cm
import numpy as np
from skvideo.io import FFmpegWriter
import tensorflow as tf
tf.enable_eager_execution()
import time

tfe = tf.contrib.eager

INNER_RADIUS = 7
OUTER_RADIUS = 3 * INNER_RADIUS
HEIGHT = 256
WIDTH = 256


def debug(a_tensor):
  print(tf.shape(a_tensor))
  print(a_tensor.dtype)


class Rules:
  #  # Birth range
  #  B1 = 0.278
  #  B2 = 0.365
  #  # Survival range
  #  D1 = 0.267
  #  D2 = 0.445
  #  # Sigmoid widths
  #  N = 0.028
  #  M = 0.147
  B1 = 0.257
  B2 = 0.336

  D1 = 0.365
  D2 = 0.549
  N = 0.028
  M = 0.147


def make_circle(size, radius):
  y, x = size
  # Get coordinate values of each point
  yy, xx = np.mgrid[:y, :x]
  # Distance between each point and the center
  lengths = np.sqrt((xx - x / 2)**2 + (yy - y / 2)**2)

  return np.clip(radius + 0.5 - lengths, 0, 1).astype(float)


def logistic2d(size, radius, roll=True, logres=None):
  """Create a circle with blurred edges

    Set roll=False to have the circle centered in the middle of the
    matrix. Default is to center at the extremes (best for convolution).
    The transition width of the blur scales with the size of the grid.
    I'm not actually sure of the math behind it, but it's what was presented
    in the code from:
    https://0fps.net/2012/11/19/conways-game-of-life-for-curved-surfaces-part-1/
    """
  y, x = size
  # Get coordinate values of each point
  yy, xx = np.mgrid[:y, :x]
  # Distance between each point and the center
  radiuses = np.sqrt((xx - x / 2)**2 + (yy - y / 2)**2)
  # Scale factor for the transition width
  if logres is None:
    logres = math.log(min(*size), 2)
  with np.errstate(over="ignore"):
    # With big radiuses, the exp overflows,
    # but 1 / (1 + inf) == 0, so it's fine
    logistic = 1 / (1 + np.exp(logres * (radiuses - radius)))
  if roll:
    logistic = np.roll(logistic, y // 2, axis=0)
    logistic = np.roll(logistic, x // 2, axis=1)
  return logistic


def sl_sigma(x, offset, width):
  return tf.sigmoid((4.0 / width) * (x - offset))


def sl_sigma2(x, start, end, width):
  return sl_sigma(x, start, width) * (1 - sl_sigma(x, end, width))


def linear_interpolation(x, start, end):
  return (end - start) * x + start


def add_speckles(count=800, intensity=1):
  field = np.zeros((HEIGHT, WIDTH), dtype=float)
  radius = int(INNER_RADIUS)
  for _ in range(count):
    r = np.random.randint(0, HEIGHT - radius)
    c = np.random.randint(0, WIDTH - radius)
    field[r:r + radius, c:c + radius] = intensity

  return field


def np_array_to_filter(ra, num_channels):
  cross_channel_couping = np.eye(3)
  cross_channel_couping = cross_channel_couping * 0.4 + 0.2

  ra /= np.sum(ra)
  ra = np.stack((ra,) * num_channels, axis=-1)
  ra = np.stack((ra,) * num_channels, axis=-1)
  return tf.convert_to_tensor((ra * np.eye(3)))


N_CHANNELS = 3
BATCH_SIZE = 1
field = np.stack([add_speckles() for _ in range(N_CHANNELS)], axis=-1)
field = tf.convert_to_tensor(
    np.reshape(field, (BATCH_SIZE, HEIGHT, WIDTH, N_CHANNELS)))

FILTER_LENGTH = OUTER_RADIUS * 2 + 1
circle = make_circle((FILTER_LENGTH,) * 2, INNER_RADIUS)
ring = make_circle((FILTER_LENGTH,) * 2, OUTER_RADIUS) - circle

circle_filter = np_array_to_filter(circle, num_channels=N_CHANNELS)
ring_filter = np_array_to_filter(ring, num_channels=N_CHANNELS)

start = time.time()
strides = tuple([1] * 4)
fps = 10
frames = 100
dt = 0.05
images_per_frame = 8
w = FFmpegWriter("smoothlife3.mp4", inputdict={"-r": str(fps)})
for frame_index in range(frames):
  frame = np.reshape(field.numpy(), (
      HEIGHT,
      WIDTH,
      N_CHANNELS,
  ))
  red = frame[:, :, 0] * frame[:, :, 1]
  green = frame[:, :, 0] * (1 - frame[:, :, 1]) * frame[:, :, 2]
  blue = frame[:, :, 0] * (1 - frame[:, :, 1]) * (1 - frame[:, :, 2])
  frame = (255 * np.stack((red, green, blue), axis=-1)).astype("uint8")
  #frame = matplotlib.colors.hsv_to_rgb(frame)
  frame = (frame * 255).astype("uint8")
  w.writeFrame(frame)

  for _ in range(images_per_frame):
    field_dot_circle = tf.nn.conv2d(
        field, circle_filter, strides=strides, padding="SAME")
    field_dot_ring = tf.nn.conv2d(
        field, ring_filter, strides=strides, padding="SAME")
    alive = sl_sigma(field_dot_circle, 0.5, Rules.M)
    transition = sl_sigma2(field_dot_ring,
                           linear_interpolation(alive, Rules.B1, Rules.D1),
                           linear_interpolation(alive, Rules.B2, Rules.D2),
                           Rules.N)
    field = field + dt * (2 * transition - 1)
    field = tf.clip_by_value(field, 0, 1)
w.close()

print("%.1f" % (time.time() - start))
