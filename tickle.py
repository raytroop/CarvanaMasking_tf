import os
import tensorflow as tf
import numpy as np
data = np.arange(12).reshape(6, 2)
x = tf.data.Dataset.from_tensor_slices(data)
x = x.batch(2)
itx = x.make_initializable_iterator()
xi = itx.get_next()
itx_init = itx.initializer

data = np.arange(12).reshape(6, 2) + 100
y = tf.data.Dataset.from_tensor_slices(data)
y = y.batch(2)
ity = y.make_initializable_iterator()
yi = ity.get_next()
ity_init = ity.initializer

def fn(feats):
    out = tf.reduce_sum(feats)
    summary = tf.summary.scalar('sum', out)
    # summary_op = tf.summary.merge_all()
    summary_op = tf.summary.merge([summary])
    return out, summary_op


outx, summ_opx = fn(xi)
print('First')
print(tf.get_collection(tf.GraphKeys.SUMMARIES))
print(tf.get_collection(tf.GraphKeys.SUMMARY_OP))
outy, summ_opy = fn(yi)
print('Second')
print(tf.get_collection(tf.GraphKeys.SUMMARIES))
print(tf.get_collection(tf.GraphKeys.SUMMARY_OP))


with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(os.path.join('logs', 'train_summaries'), sess.graph)
    eval_writer = tf.summary.FileWriter(os.path.join('logs', 'eval_summaries'), sess.graph)
    for e in range(2):
        sess.run(itx_init)
        for i in range(3):
            outx_, summ = sess.run([outx, summ_opx])
            print(i, outx_)
            train_writer.add_summary(summ, e*3+i)

        sess.run(ity_init)
        for i in range(3):
            # outy_ = sess.run(outy)
            outy_, summ = sess.run([outy, summ_opy])
            print(i, outy_)
            eval_writer.add_summary(summ, e*3+i)