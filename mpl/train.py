import click
import numpy as np
import sklearn.datasets
import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import plot_decision_boundary

tf.compat.v1.disable_eager_execution()

NUM_CLASSES = 2


def cross_entropy(target, prob, eps=1e-8):
    loss = -tf.reduce_sum(target * tf.math.log(prob + eps), -1)

    return loss


def linear(input, weight, bias):
    input = tf.matmul(input, weight)
    input = tf.nn.bias_add(input, bias)

    return input


def model(input, params):
    input = linear(input, params[0], params[1])
    input = tf.nn.relu(input)
    input = linear(input, params[2], params[3])
    # input = tf.nn.relu(input)
    # input = linear(input, params[4], params[5])
    input = tf.nn.softmax(input, -1)

    return input


def build_graph(lr, w, tw):
    student = model
    teacher = model

    # inputs ###########################################################################################################

    inputs = {
        'x_s': tf.compat.v1.placeholder(tf.float32, (None, 2)),
        'y_s': tf.compat.v1.placeholder(tf.int64, (None,)),
        'x_u': tf.compat.v1.placeholder(tf.float32, (None, 2)),
    }

    # weights init #####################################################################################################

    theta = []
    for f_in, f_out in [(2, 32), (32, 2)]:
        theta.append(tf.Variable(tf.random.truncated_normal((f_in, f_out), 0., 0.1)))
        theta.append(tf.Variable(tf.zeros((f_out,))))

    psi = []
    for f_in, f_out in [(2, 32), (32, 2)]:
        psi.append(tf.Variable(tf.random.truncated_normal((f_in, f_out), 0., 0.1)))
        psi.append(tf.Variable(tf.zeros((f_out,))))

    # student ##########################################################################################################

    loss_student = [
        cross_entropy(target=teacher(inputs['x_u'], psi), prob=student(inputs['x_u'], theta)),
    ]
    loss_student = w * sum(tf.reduce_mean(l) for l in loss_student)

    theta_grad = tf.gradients(loss_student, theta)
    theta_prime = [t - lr * t_g for t, t_g in zip(theta, theta_grad)]
    student_update = tf.group([t.assign(t_p) for t, t_p in zip(theta, theta_prime)])

    # teacher ##########################################################################################################

    loss_teacher = [
        tw * cross_entropy(target=tf.one_hot(inputs['y_s'], 2), prob=student(inputs['x_s'], theta_prime)),
        (1 - tw) * cross_entropy(target=tf.one_hot(inputs['y_s'], 2), prob=teacher(inputs['x_s'], psi)),
    ]
    loss_teacher = (1 - w) * sum(tf.reduce_mean(l) for l in loss_teacher)

    psi_grad = tf.gradients(loss_teacher, psi)
    psi_prime = [p - lr * p_g for p, p_g in zip(psi, psi_grad)]
    teacher_update = tf.group([p.assign(p_p) for p, p_p in zip(psi, psi_prime)])

    # evaluation #######################################################################################################

    class_index = tf.argmax(student(inputs['x_s'], theta), -1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(
        inputs['y_s'],
        class_index,
    ), tf.float32))

    # outputs ##########################################################################################################

    outputs = {
        'loss_student': loss_student,
        'loss_teacher': loss_teacher,
        'student_update': student_update,
        'teacher_update': teacher_update,
        'accuracy': accuracy,
        'class_index': class_index,
    }

    return inputs, outputs


@click.command()
@click.option('--experiment-path', type=click.Path(), required=True)
def main(experiment_path):
    ins, outs = build_graph(lr=1e-1, w=0.5, tw=0.5)

    u_size = 6
    x_u, _ = sklearn.datasets.make_moons(2000, noise=0.1)
    x_s, y_s = sklearn.datasets.make_moons(u_size, noise=0.1)
    x_val, y_val = sklearn.datasets.make_moons(2000, noise=0.1)

    writer = SummaryWriter(experiment_path)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for i in tqdm(range(1, 10000 + 1)):
            l_s, l_t, _, _ = sess.run(
                [outs['loss_student'], outs['loss_teacher'], outs['student_update'], outs['teacher_update']],
                feed_dict={
                    ins['x_u']: x_u[np.random.choice(x_u.shape[0], u_size, replace=False)],
                    ins['x_s']: x_s,
                    ins['y_s']: y_s,
                })

            if i % 100 == 0:
                acc = sess.run(outs['accuracy'], feed_dict={
                    ins['x_s']: x_val,
                    ins['y_s']: y_val,
                })

                writer.add_scalar('loss_student', l_s, global_step=i)
                writer.add_scalar('loss_teacher', l_t, global_step=i)
                writer.add_scalar('accuracy', acc, global_step=i)
                fig = plot_decision_boundary(
                    x_s, y_s, lambda x: sess.run(outs['class_index'], feed_dict={ins['x_s']: x}))
                writer.add_figure('sup-surface', fig, global_step=i)
                fig = plot_decision_boundary(
                    x_val, y_val, lambda x: sess.run(outs['class_index'], feed_dict={ins['x_s']: x}))
                writer.add_figure('val-surface', fig, global_step=i)
    writer.flush()


if __name__ == '__main__':
    main()
