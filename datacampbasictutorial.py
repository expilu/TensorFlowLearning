import tensorflow as tf

# Initialize two constants
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

print(result) # no calculation done here, we just defined the model

config = tf.ConfigProto(log_device_placement=True) # this shows which device is used for each operation

with tf.Session(config = config) as sess:
    output = sess.run(result)
    print(output) # here it is calculated
