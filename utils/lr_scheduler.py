import tensorflow as tf
import math

def cosine_schedule(base_lr, total_steps, warmup_steps ):
    def step_fn(epoch):
        lr = base_lr

        progress = (epoch - warmup_steps) / float(total_steps -  warmup_steps)

        progress = tf.clip_by_value(progress, 0.0, 1.0)

        lr = lr * 0.5 * (1.0 + tf.cos(math.pi * progress))
        
        if warmup_steps:
            lr = lr * tf.minimum(1.0 , epoch/warmup_steps)
        
        return lr
    

    return step_fn

def linear_scheduler(dim_embed, base_lr, total_steps, warmup_steps):
    def step_fn(epoch):
        return dim_embed**(-0.5) * min(total_steps**(-0.5), step * warmup_steps**(-1.5))
