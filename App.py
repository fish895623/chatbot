from dotenv import load_dotenv

load_dotenv()
import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # need to disable eager in TF2.x
# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b

# Launch the graph in a session.
sess = tf.compat.v1.Session()

# Evaluate the tensor `c`.
print(sess.run(c))  # prints 30.0
print('helloWorld')
