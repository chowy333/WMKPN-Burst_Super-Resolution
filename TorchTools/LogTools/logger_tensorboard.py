# import tensorflow as tf
from torch.utils.tensorboard import SummaryWriter
import numpy as np

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x


class Tf_Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        # self.writer = tf.summary.create_file_writer(log_dir)
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)
        # with self.writer.as_default():
        #     tf.summary.scalar(tag, value, step)
            # self.writer.flush()

    def image_summary(self, tag, images, step):
        """Log a list of images."""
        self.writer.add_images(tag, images, step)


        #for i, img in enumerate(images):
        #    if len(img.shape) == 2:
        #        img = img[np.newaxis, :,:]
        #    elif len(img.shape) == 3:
        #        print(img.shape)
        #        pass
        #        # img = img[np.newaxis,:,:,:]

        ## with self.writer.as_default():
        #     for i, img in enumerate(images):
        #         if len(img.shape) == 2:
        #             img = img[np.newaxis,:,:,np.newaxis]
        #         elif len(img.shape) == 3:
        #             img = img[np.newaxis,:,:,:]
        #         tf.summary.image('%s/%d' % (tag, i), img, step)
            # self.writer.flush()

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values**2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        self.writer.add_histogram(tag, hist, step)
        # with self.writer.as_default():
        #     tf.summary.histogram(tag, hist, step)
        #     self.writer.flush()
