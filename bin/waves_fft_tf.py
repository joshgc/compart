import argparse
import collections
import itertools
import math
from matplotlib import cm
import numpy as np
from skvideo.io import FFmpegWriter
from skvideo.io import ffmpeg
from scipy.ndimage import filters
import shlex
import subprocess
from PIL import Image
import tensorflow as tf
tf.enable_eager_execution()
import time


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--job-dir",
      dest="job_dir",
      default="gs://joshgc-tpu-art-ml/sl/tf/local",
      help="Inherited from CMLE")

  return parser.parse_args()


for key in [
    "_FFMPEG_APPLICATION", "_FFMPEG_PATH", "_FFMPEG_SUPPORTED_DECODERS",
    "_FFMPEG_SUPPORTED_ENCODERS", "_HAS_FFMPEG"
]:
  print("{}={}".format(key, getattr(ffmpeg, key, None)))

_USE_FFMPEG = ffmpeg._HAS_FFMPEG == 1
print("Using FFMPEG={}".format(_USE_FFMPEG))


def debug(a_tensor):
  print(tf.shape(a_tensor))
  print(a_tensor.dtype)


class FieldFilter(object):

  def __call__(self, field):
    raise NotImplemented


class FieldVarianceFilter(FieldFilter):

  def __init__(self, threshold):
    self.threshold = threshold

  def __call__(self, field):
    if np.var(field) > self.threshold:
      return True
    print("Variance too small")
    return False


class DifferenceFilter(FieldFilter):

  def __init__(self, num_frames_later, threshold):
    self.num_frames_later = num_frames_later
    self.threshold = threshold
    self.deque = collections.deque([], num_frames_later + 1)

  def __call__(self, field):
    self.deque.append(field)
    if len(self.deque) < self.num_frames_later + 1:
      return True
    old_field = self.deque.popleft()
    if len(self.deque) > self.num_frames_later:
      raise ValueError("Too many elements in deque")
    if np.max(np.abs(field - old_field)) > self.threshold:
      return True
    print("{} diff too small".format(self.num_frames_later))
    return False


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
  return make_ellipsis(size, radius, radius)


def ellipsis_mask(size, center_y, center_x, axis_y, axis_x):
  yy, xx = np.mgrid[:size[0], :size[1]]
  axis_x_2 = float(axis_x**2)
  axis_y_2 = float(axis_y**2)
  weighted_distance = np.square(xx - center_x) / axis_x_2 + np.square(
      yy - center_y) / axis_y_2
  return np.where(weighted_distance >= 1, 0, 1)


def make_ellipsis(size, axis_y, axis_x):
  conv_weights = ellipsis_mask(size, size[0] / 2, size[1] / 2, axis_y,
                               axis_x).astype(np.float32)
  return filters.gaussian_filter(conv_weights, sigma=1)


def left_right_circle(size, radius):
  y, x = size
  yy, xx = np.mgrid[:y, :x]
  logres = math.log(min(*size), 2)

  radiuses = np.sqrt(np.square(xx - x / 2) + np.square(yy - y / 2))
  with np.errstate(over="ignore"):
    logistic = 1 / (1 + np.exp(logres * (radiuses - radius)))
  logistic *= np.where(xx > x / 2, -1, 3)
  print(logistic.shape)
  return logistic


def sl_sigma(x, offset, width):
  return tf.sigmoid((4.0 / width) * (x - offset))


def sl_sigma2(x, start, end, width):
  return sl_sigma(x, start, width) * (1 - sl_sigma(x, end, width))


def linear_interpolation(x, start, end):
  return (end - start) * x + start


def add_speckles(field, radius, density=800 / 256.0 / 256, intensity=1):
  height, width = field.shape
  for _ in range(int(density * height * width)):
    r = np.random.randint(0, height - radius)
    c = np.random.randint(0, width - radius)
    field[r:r + radius, c:c + radius] = intensity

  return field


def normalize_and_fft(conv_filter):
  conv_tf = tf.convert_to_tensor(
      conv_filter / np.sum(conv_filter), dtype=np.float32)
  return tf.spectral.rfft2d(conv_tf)


def make_conv_a_fft(size, outer_limit):
  circle = make_circle(size, outer_limit)
  return normalize_and_fft(circle)


def make_conv_b_fft(size, inner_limit, outer_limit):
  ring = make_circle(size, outer_limit) - make_circle(size, inner_limit)
  return normalize_and_fft(ring)


def evolution(evolution_rules):
  INNER_RADIUS = 7
  OUTER_RADIUS = 3 * INNER_RADIUS
  HEIGHT = 1024
  WIDTH = int(256 * 16.0 / 9 / 2) * 2
  dt = 0.02
  SIZE = (HEIGHT, WIDTH)

  conv_a_fft = make_conv_a_fft(SIZE, INNER_RADIUS)
  conv_b_fft = make_conv_b_fft(SIZE, INNER_RADIUS, OUTER_RADIUS)

  field = np.zeros(SIZE, dtype=float)
  field = add_speckles(field, INNER_RADIUS)
  field = tf.convert_to_tensor(field, dtype=np.float32)

  while True:
    yield field
    field_fft = tf.spectral.rfft2d(field)
    field_dot_circle = tf.spectral.irfft2d(field_fft * conv_a_fft)
    field_dot_ring = tf.spectral.irfft2d(field_fft * conv_b_fft)
    alive = sl_sigma(field_dot_circle, 0.50, Rules.M)

    transition = sl_sigma2(field_dot_ring,
                           linear_interpolation(alive, Rules.B1, Rules.D1),
                           linear_interpolation(alive, Rules.B2, Rules.D2),
                           Rules.N)
    field = field + dt * (2 * transition - 1)
    field = tf.clip_by_value(field, 0, 1)


def run_simulation(rules):
  field_gen = evolution(None)

  fps = 15
  frames = 300

  if _USE_FFMPEG:
    w = FFmpegWriter("smoothlife.mp4", inputdict={"-r": str(fps)})
  else:
    w = None

  gen = evolution(rules)
  #gen = itertools.takewhile(FieldVarianceFilter(1e-7), gen)
  gen = itertools.takewhile(DifferenceFilter(1, 0.01), gen)
  gen = itertools.takewhile(DifferenceFilter(2, 0.01), gen)

  start = time.time()
  for i, field in enumerate(field_gen):
    if i > frames:
      break

    np_field = field.numpy()
    frame = cm.viridis(np_field)
    frame *= 255
    frame = frame.astype("uint8")
    if _USE_FFMPEG:
      w.writeFrame(frame)
    else:
      image_bytes = Image.fromarray(frame[:, :, 0:3], "RGB")
      image_bytes.save("frame_{:09d}.jpg".format(i))

  if w:
    w.close()
  print("%.1f" % (time.time() - start))


args = parse_args()
run_simulation(None)
subprocess.call(
    shlex.split("gsutil -m cp smoothlife.mp4 frame_0*.jpg {}".format(
        args.job_dir)))
