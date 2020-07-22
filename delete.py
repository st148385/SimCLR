import tensorflow as tf
import numpy as np
import tensorflow.keras
import tensorflow_datasets as tfds
import gin
from PIL import Image
import matplotlib.pyplot as plt
from math import pi as pi
from tensorflow_core.python.ops.gen_image_ops import sample_distorted_bounding_box_v2

# import tensorflow as tf
# strategy = tf.distribute.MirroredStrategy()
# print ('Number of devices: {}'.format(strategy.num_replicas_in_sync))

def get_negative_mask(batch_size):
    # return a mask that removes the similarity score of equal/similar images.
    # this function ensures that only distinct pair of images get their similarity scores
    # passed as negative examples
    negative_mask = np.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0
    return tf.constant(negative_mask)

def sthalles_loss(z_i, z_j, tau=0.5, batch_size=16):
    z_i = tf.math.l2_normalize(z_i, axis=1)
    z_j = tf.math.l2_normalize(z_j, axis=1)
    l_pos = tf.matmul(tf.expand_dims(z_i, 1), tf.expand_dims(z_j, 2))
    l_pos = tf.reshape(l_pos, (batch_size, 1))
    l_pos = l_pos/tau
    negatives = tf.concat([z_j, z_i], axis=0)
    loss = 0
    for positives in [z_i, z_j]:
        l_neg = tf.tensordot(tf.expand_dims(positives, 1), tf.expand_dims(tf.transpose(negatives), 0), axes=2)
        l_neg = tf.boolean_mask(l_neg, get_negative_mask(batch_size))
        l_neg = tf.reshape(l_neg, (batch_size, -1))
        l_neg = l_neg / tau
        logits = tf.concat([l_pos, l_neg], axis=1)
        labels = tf.zeros(batch_size, dtype=tf.int32)
        loss += tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
                                                              reduction=tf.keras.losses.Reduction.SUM)(y_pred=logits, y_true=labels)
    loss = loss / (2 * batch_size)
    return loss

def anotherLossImplementation(a, b, tau):
    a=tf.reshape(a, (-1,1))
    b=tf.reshape(b, (-1,1))
    a_norm = tf.reshape(tf.math.l2_normalize(a, axis=1), shape=(-1, 1))
    a_hat = tf.math.divide(a, a_norm)                               # â = a / |a|
    b_norm = tf.reshape(tensor=tf.math.l2_normalize(b, axis=1), shape=(-1, 1))
    b_hat = tf.math.divide(b, b_norm)
    a_hat_b_hat = tf.concat([a_hat, b_hat], axis=0)                 #All augmented images i and all images j in one vector => (BS*128*2, 1)
    a_hat_b_hat_transpose = tf.transpose(a_hat_b_hat)               #(1, BS*128*2)
    b_hat_a_hat = tf.concat([b_hat, a_hat], axis=0)
    sim = tf.matmul(a_hat_b_hat, a_hat_b_hat_transpose)             #(BS*128*2, BS*128*2)
    sim_over_tau = tf.math.divide(sim, tau)
    exp_sim_over_tau = tf.exp(sim_over_tau)                         #(BS*128*2, BS*128*2)
    exp_sim_over_tau_diag = tf.linalg.diag_part(exp_sim_over_tau)   #(BS*128*2,) This is the sim with k=i, so sim(z_i,z_i)
    sum_of_rows = tf.reduce_sum(exp_sim_over_tau, axis=1)           #sum of all elements in every one line -> (BS*128*2,)
    denominators = sum_of_rows - exp_sim_over_tau_diag              #Remove the e^(sim(zi,zi)/tau) from the sum -> Denominator/Nenner ready (4096,)

    cosine_sim = tf.keras.losses.CosineSimilarity()(a_hat_b_hat, b_hat_a_hat)   #sim( [concat(â+ĉ),1] , [concat(ĉ+â),1] )  ->  <(N,1),(1,N)>  ->  scalar
    cosine_sim_over_tau = tf.math.divide(cosine_sim, tau)
    numerators = tf.exp(cosine_sim_over_tau)                        #Numerator/Zähler ready (scalar)

    numerator_over_denominator = tf.math.divide(numerators, denominators)   #Last steps are: mean(-log(numerator/denominator))
    negativelog_num_over_den = -tf.math.log(numerator_over_denominator)
    return tf.math.reduce_mean(negativelog_num_over_den)

tau=0.5
a=tf.random.uniform(shape=(16,128), minval=0, maxval=1)
b=tf.random.uniform(shape=(16,128), minval=0, maxval=1)
c=a
d=b
loss_obv=anotherLossImplementation(a,b,tau)

loss_sthalles=sthalles_loss(c,d)


"""
Notizen zu - in SimCLR paper - verwendetem Projection Head:

    elif FLAGS.proj_head_mode == 'nonlinear':                          #nonlinear is what we use
       for j in range(FLAGS.num_proj_layers):                          #FLAGS.num_proj_layers = 3 for SimCLRv2
         if j != FLAGS.num_proj_layers - 1:                            #If j is NOT the final dense-layer: Do ...
           # for the middle layers, use bias and relu for the output.
           dim, bias_relu = mid_dim, True                              #All dense-layers BUT the final one, use relu with bias
         else:
           # for the final layer, neither bias nor relu is used.       #Just for the final dense-layer: 
           dim, bias_relu = FLAGS.proj_out_dim, False                  #  Don't use batch norm or relu-activation
         hiddens = linear_layer(
             hiddens, is_training, dim,                                #All linear_layers (which are the 3 or 2 Dense-Layers of g(•)) use "dim"
             use_bias=bias_relu, use_bn=True, name='nl_%d'%j)          #as their number of neurons. Due to the flag FLAGS.proj_out_dim=128 
         hiddens = tf.nn.relu(hiddens) if bias_relu else hiddens       #every single one of the dense-layers have 128 neurons on Google's github.
         hiddens_list.append(hiddens)
"""



class test_lr(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
        with self.warmup_steps = tf.math.ceil(overallSteps * 0.1):
        warmup phase takes 10% of all training steps.

        If this is unwanted, also change N by solving:
        solve cos( 2pi*(K*Warmup-Warmup) / N ) = 0 ,N       (K is 1/warmupPercent, which looks for "f(x=finalStep)=0")
        i.e. K = 10 for 10% warmup-steps of overallSteps, K = 20 for 5% warmup-steps of overallSteps

        Example: For 20% warmupSteps of overallSteps:
        solve cos( 2pi*(5*Warmup-Warmup) / N ) = 0 ,N -> N = 16*W, so change N=16 and warmupPercent=0.2
        For 5% warmupSteps of overallSteps:
        solve cos( 2pi*(20*Warmup-Warmup) / N ) = 0 ,N -> N = 76*W, so change N=76 and warmupPercent=0.05
    """
    def __init__(self, lr_max, warmup_steps=20000):
        super(test_lr, self).__init__()

        self.lr_max = lr_max
        self.lr_max = tf.cast(self.lr_max, tf.float32)

        self.overallSteps = (100000//512) * 100

        self.warmup_steps = tf.math.ceil(self.overallSteps * 0.05)

    def __call__(self, step):

        #solve cos(2pi*(10*W-W) / (N) ) = 0 ,N  -> N=36
        cos_decay =  (tf.math.cos( ( 2*pi*(step-self.warmup_steps) ) / (76*(self.warmup_steps)) ))


        lin_warmup = step * (self.warmup_steps ** -1)

        return abs( (self.lr_max) * tf.math.minimum(cos_decay, lin_warmup) )
        #return tf.math.rsqrt(self.d_model) * tf.math.minimum(cos_decay, lin_warmup)


teste_learning_rate_schedule = test_lr(lr_max=0.001)    #actual_lr_max ≈ +-5% * lr_max

plt.plot(teste_learning_rate_schedule(tf.range(195*100, dtype=tf.float32)))
plt.ylabel("Learning Rate")
plt.xlabel("Train Step")
plt.show()


'''#####
######
import os

path = r"C:\\Users\Mari\PycharmProjects\experiments\models"
os.chdir(path)
files = sorted(os.listdir(os.getcwd()), key=os.path.getmtime)

oldest = files[0]
newest = files[-1]

print("Oldest:", oldest)
print("Newest:", newest)
print("All by modified oldest to newest:", files)
######


tf.print(np.array([[1,5],[2,1]]))

x=51,1
#y=tf.keras.layers.Dense(240)(x)
#tf.print(y)

#print(tfds.list_builders())

c = tf.constant([1.0, 2.0])
c=tf.math.l2_normalize(c)       #Vektor c / sqrt(1**2 + 2**2) = [[1,2]]/sqrt(5) = [[1/sqrt(5),2/sqrt(5)]] = [[0.4472,0.8944]]
print(c,"\n")

d = tf.constant( [ [[1.0, 2.0],[5,-1]], [[1,2],[5,-1]] ] )
print( tf.math.l2_normalize(d, axis=1) )


def crop_and_resize(image, height, width):
    """Make a random crop and resize it to height `height` and width `width`.

  Args:
    image: Tensor representing the image.
    height: Desired image height.
    width: Desired image width.

  Returns:
    A `height` x `width` x channels Tensor holding a random crop of `image`.
  """
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])     #Also ist bbox ein Viereck mit (ymin,xmin,ymax,xmax)=(0,0,1,1)
    aspect_ratio = width / height

    begin,size,bbox_for_draw = sample_distorted_bounding_box_v2(
        image_size=tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=0.1, #Mindestens N% des Originalbilds müssen sich in der cropped version wiederfinden lassen
        aspect_ratio_range=(3. / 4 * aspect_ratio, 4. / 3. * aspect_ratio), #cropped area hat shape im Bereich [0.75*width/height, 1.333*width/height]
        area_range=(0.01,0.2),  #cropped area muss in diesem Bereich des Originalbilds liegen (Werte von SimCLR-Github, obwohl nur >0.1 als untere Grenze Sinn macht?)
        max_attempts=100    #Nach 100 tries einfach das Originalbild beibehalten
    )
    slice_of_image=tf.slice(image, begin, size)
    return tf.image.resize(slice_of_image, size=(width,height), method='bicubic')

#crop_and_resize
image = Image.open('C:\\Users\Mari\Pictures\Hopetoun Falls Wasserfall Waterfall.jpg')
image = tf.cast(np.array(image), tf.float32) / 255.0
plt.subplot(2,1,1)
plt.imshow(image)

slice=crop_and_resize(image,1200,1200)
plt.subplot(2,1,2)
plt.imshow(slice)
plt.show()
'''####


'''
J1 = tf.ones( (128,1) )
J2 = tf.ones( (128,1) )

print(tf.concat( (J1,J2), axis=0 ))
print(tf.concat( (J1,J2), axis=1 ))
'''
'''
#AssertionError:
meineliste = ['element']    #meineliste: ['element']

assert len(meineliste) >= 1
meineliste.pop()    #meineliste: []

assert len(meineliste) >= 1
#-> AssertionError, weil len(meineliste) < 1, wodurch [len(meineliste) >= 1]=False
'''

#"@tf.function" allows for the intuitive use of both eager execution and AutoGraph, whereby a function can be run using Python syntax initially
#and #then transferred into the equivalent graph code.

#
#3) Was ist yaml? Siehe to_do in utils_misc
'''
#dict = {'note mathe':1,'note fach':2}

#print(dict['note mathe'])
'''

#_fkt_name für private
#dataset.map(funktion) führt funktion auf jedes einzelne element von dataset aus
'''
import tensorflow as tf
import datetime
import os
import yaml
import time
import shutil

print(tf.__version__)
from models.cnn_small import SmallCNN
from models.resnet_simclr import ResNetSimCLR
from utils.losses import _dot_simililarity_dim1 as sim_func_dim1, _dot_simililarity_dim2 as sim_func_dim2
from utils.helpers import get_negative_mask, gaussian_filter
from augmentation.transforms import read_images, distort_simclr, read_record, distort_with_rand_aug

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

config = yaml.load(open("./config.yaml", "r"), Loader=yaml.FullLoader)
input_shape = eval(config['input_shape'])

train_dataset = tf.data.TFRecordDataset('./data/tfrecords/train.tfrecords')
train_dataset = train_dataset.map(lambda x: read_record(x, input_shape),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(distort_simclr, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.map(gaussian_filter, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.repeat(config['epochs'])
train_dataset = train_dataset.shuffle(4096)
train_dataset = train_dataset.batch(config['batch_size'], drop_remainder=True)
train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

criterion = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(3e-4)

# model = SmallCNN(out_dim=config['out_dim'])
model = ResNetSimCLR(input_shape=input_shape, out_dim=config['out_dim'])

# Mask to remove positive examples from the batch of negative samples
negative_mask = get_negative_mask(config['batch_size'])

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = os.path.join('logs', current_time, 'train')
train_summary_writer = tf.summary.create_file_writer(train_log_dir)


@tf.function
def train_step(xis, xjs):
    with tf.GradientTape() as tape:
        ris, zis = model(xis)
        rjs, zjs = model(xjs)

        # normalize projection feature vectors
        zis = tf.math.l2_normalize(zis, axis=1)
        zjs = tf.math.l2_normalize(zjs, axis=1)

        # tf.summary.histogram('zis', zis, step=optimizer.iterations)
        # tf.summary.histogram('zjs', zjs, step=optimizer.iterations)

        l_pos = sim_func_dim1(zis, zjs)
        l_pos = tf.reshape(l_pos, (config['batch_size'], 1))
        l_pos /= config['temperature']
        # assert l_pos.shape == (config['batch_size'], 1), "l_pos shape not valid" + str(l_pos.shape)  # [N,1]

        negatives = tf.concat([zjs, zis], axis=0)

        loss = 0

        for positives in [zis, zjs]:
            l_neg = sim_func_dim2(positives, negatives)

            labels = tf.zeros(config['batch_size'], dtype=tf.int32)

            l_neg = tf.boolean_mask(l_neg, negative_mask)
            l_neg = tf.reshape(l_neg, (config['batch_size'], -1))
            l_neg /= config['temperature']

            # assert l_neg.shape == (
            #     config['batch_size'], 2 * (config['batch_size'] - 1)), "Shape of negatives not expected." + str(
            #     l_neg.shape)
            logits = tf.concat([l_pos, l_neg], axis=1)  # [N,K+1]
            loss += criterion(y_pred=logits, y_true=labels)

        loss = loss / (2 * config['batch_size'])
        tf.summary.scalar('loss', loss, step=optimizer.iterations)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))


with train_summary_writer.as_default():
    for xis, xjs in train_dataset:
        # print(tf.reduce_min(xis), tf.reduce_max(xjs))
        # fig, axs = plt.subplots(nrows=2, ncols=2, constrained_layout=False)
        # axs[0, 0].imshow(xis[0])
        # axs[0, 1].imshow(xis[1])
        # axs[1, 0].imshow(xis[2])
        # axs[1, 1].imshow(xis[3])
        # plt.show()
        # start = time.time()
        train_step(xis, xjs)
        # end = time.time()
        # print("Total time per batch:", end - start)

model_checkpoints_folder = os.path.join(train_log_dir, 'checkpoints')
if not os.path.exists(model_checkpoints_folder):
    os.makedirs(model_checkpoints_folder)
    shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))

model.save_weights(os.path.join(model_checkpoints_folder, 'model.h5'))
'''