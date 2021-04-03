import click
import numpy as np
import sklearn.datasets
import tensorflow as tf
from tensorboardX import SummaryWriter
from tqdm import tqdm

from utils import plot_decision_boundary

tf.compat.v1.disable_eager_execution()

NUM_CLASSES = 2


def ema(input, a):
    prev = tf.Variable(tf.zeros_like(input))

    return a * prev + (1 - a) * input


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
    input = tf.nn.softmax(input, -1)

    return input


def build_graph(lr, w, tw):
    student = model
    teacher = model

    # inputs ###########################################################################################################

    inputs = {
        "x_s": tf.compat.v1.placeholder(tf.float32, (None, 2)),
        "y_s": tf.compat.v1.placeholder(tf.int64, (None,)),
        "x_u": tf.compat.v1.placeholder(tf.float32, (None, 2)),
    }

    # weights init #####################################################################################################

    theta = []
    for f_in, f_out in [(2, 32), (32, 2)]:
        theta.append(tf.Variable(tf.random.truncated_normal((f_in, f_out), 0.0, 0.1)))
        theta.append(tf.Variable(tf.zeros((f_out,))))

    psi = []
    for f_in, f_out in [(2, 32), (32, 2)]:
        psi.append(tf.Variable(tf.random.truncated_normal((f_in, f_out), 0.0, 0.1)))
        psi.append(tf.Variable(tf.zeros((f_out,))))

    # student ##########################################################################################################

    loss_student = [
        cross_entropy(target=teacher(inputs["x_u"], psi), prob=student(inputs["x_u"], theta)),
        # 1e-7 * sum(tf.reduce_sum(t**2) for t in theta),
    ]
    loss_student = w * sum(tf.reduce_mean(l) for l in loss_student)

    theta_grad = tf.gradients(loss_student, theta)
    theta_grad = [ema(t_g, 0.9) for t_g in theta_grad]
    theta_prime = [t - lr * t_g for t, t_g in zip(theta, theta_grad)]
    student_update = tf.group([t.assign(t_p) for t, t_p in zip(theta, theta_prime)])

    # teacher ##########################################################################################################

    loss_teacher = [
        tw
        * cross_entropy(
            target=tf.one_hot(inputs["y_s"], 2), prob=student(inputs["x_s"], theta_prime)
        ),
        (1 - tw)
        * cross_entropy(target=tf.one_hot(inputs["y_s"], 2), prob=teacher(inputs["x_s"], psi)),
        # 1e-3 * sum(tf.reduce_sum(p**2) for p in psi),
    ]
    loss_teacher = (1 - w) * sum(tf.reduce_mean(l) for l in loss_teacher)

    psi_grad = tf.gradients(loss_teacher, psi)
    psi_grad = [ema(p_g, 0.9) for p_g in psi_grad]
    psi_prime = [p - lr * p_g for p, p_g in zip(psi, psi_grad)]
    teacher_update = tf.group([p.assign(p_p) for p, p_p in zip(psi, psi_prime)])

    # evaluation #######################################################################################################

    teacher_outputs = {"class_index": tf.argmax(teacher(inputs["x_s"], psi), -1)}
    teacher_outputs["accuracy"] = tf.reduce_mean(
        tf.cast(
            tf.equal(
                inputs["y_s"],
                teacher_outputs["class_index"],
            ),
            tf.float32,
        )
    )

    student_outputs = {"class_index": tf.argmax(student(inputs["x_s"], theta), -1)}
    student_outputs["accuracy"] = tf.reduce_mean(
        tf.cast(
            tf.equal(
                inputs["y_s"],
                student_outputs["class_index"],
            ),
            tf.float32,
        )
    )

    # outputs ##########################################################################################################

    outputs = {
        "loss_student": loss_student,
        "loss_teacher": loss_teacher,
        "student_update": student_update,
        "teacher_update": teacher_update,
        "teacher": teacher_outputs,
        "student": student_outputs,
    }

    return inputs, outputs


@click.command()
@click.option("--experiment-path", type=click.Path(), required=True)
def main(experiment_path):
    ins, outs = build_graph(lr=1e-1, w=0.5, tw=0.5)

    s_size = 6
    u_size = s_size
    x_s, y_s = sklearn.datasets.make_moons(s_size, noise=0.1)
    x_u, _ = sklearn.datasets.make_moons(2000, noise=0.1)
    x_val, y_val = sklearn.datasets.make_moons(2000, noise=0.1)

    writer = SummaryWriter(experiment_path)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for i in tqdm(range(1, 100000 + 1)):
            loss_stu, loss_tea, _, _ = sess.run(
                [
                    outs["loss_student"],
                    outs["loss_teacher"],
                    outs["student_update"],
                    outs["teacher_update"],
                ],
                feed_dict={
                    ins["x_u"]: x_u[np.random.choice(x_u.shape[0], u_size, replace=False)],
                    ins["x_s"]: x_s,
                    ins["y_s"]: y_s,
                },
            )

            if i % 100 == 0:
                acc_tea, acc_stu = sess.run(
                    [outs["teacher"]["accuracy"], outs["student"]["accuracy"]],
                    feed_dict={
                        ins["x_s"]: x_val,
                        ins["y_s"]: y_val,
                    },
                )

                writer.add_scalar("teacher/loss", loss_tea, global_step=i)
                writer.add_scalar("student/loss", loss_stu, global_step=i)
                writer.add_scalar("teacher/accuracy", acc_tea, global_step=i)
                writer.add_scalar("student/accuracy", acc_stu, global_step=i)

                fig = plot_decision_boundary(
                    x_s,
                    y_s,
                    lambda x: sess.run(outs["teacher"]["class_index"], feed_dict={ins["x_s"]: x}),
                )
                writer.add_figure("teacher/sup", fig, global_step=i)
                fig = plot_decision_boundary(
                    x_val,
                    y_val,
                    lambda x: sess.run(outs["teacher"]["class_index"], feed_dict={ins["x_s"]: x}),
                )
                writer.add_figure("teacher/val", fig, global_step=i)
                fig = plot_decision_boundary(
                    x_s,
                    y_s,
                    lambda x: sess.run(outs["student"]["class_index"], feed_dict={ins["x_s"]: x}),
                )
                writer.add_figure("student/sup", fig, global_step=i)
                fig = plot_decision_boundary(
                    x_val,
                    y_val,
                    lambda x: sess.run(outs["student"]["class_index"], feed_dict={ins["x_s"]: x}),
                )
                writer.add_figure("student/val", fig, global_step=i)
    writer.flush()


if __name__ == "__main__":
    main()
