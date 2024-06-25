import os
import sys

sys.path.append(os.getcwd())

import tensorflow as tf
import numpy as np
import src.model_code.model_persistence as model_persistence
import src.util.constants as constants
import src.model_code.data_loader as data_loader


loader = data_loader.DataLoader()

class Distiller(tf.keras.Model):
    def __init__(self, student, teacher):
        super().__init__()
        self.teacher = teacher
        self.student = student

    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha=0.1,
            temperature=3,
    ):
        """Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super().compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(
            self, x=None, y=None, y_pred=None, sample_weight=None, allow_empty=False
    ):
        teacher_pred = self.teacher(x, training=False)
        student_loss = self.student_loss_fn(y, y_pred)

        distillation_loss = self.distillation_loss_fn(
            tf.keras.activations.softmax(teacher_pred / self.temperature, axis=1), # evtl tf.nn.softmax
            tf.keras.activations.softmax(y_pred / self.temperature, axis=1),
        ) * (self.temperature ** 2)

        loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss
        return loss

    def call(self, x):
        return self.student(x)


# Create the student
student = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(8, 3, activation='relu'),
        tf.keras.layers.Conv2D(8, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.Conv2D(16, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.Conv2D(32, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.Conv2D(64, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.Conv2D(128, 3, activation='relu'),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer='l2'),
        tf.keras.layers.Dense(constants.NUM_CLASSES, activation='softmax')
    ],
    name="student",
)

# Clone student for later comparison
student_scratch = tf.keras.models.clone_model(student)

teacher = model_persistence.model_load("perfect_wolf_10_3_20240611_143102.keras")


# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=tf.keras.optimizers.Adam(),
    metrics=["accuracy"],
    student_loss_fn=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=tf.keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
)

# Distill teacher to student
distiller.fit(
    loader.load_training_data(),
    validation_data=loader.load_validation_data(),
    epochs=2,
    callbacks=[model_persistence.model_checkpoint_callback()]
)

# Evaluate student on test dataset
distiller.evaluate(loader.load_validation_data())

# save student
model_persistence.model_save_sequential(student, "perfect_wolf_knowledge_distillation_student_1.keras")

# Train student as done usually
student_scratch.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

# save distiller
model_persistence.model_save_sequential(distiller, "perfect_wolf_knowledge_distillation_distiller_1.keras")


# Train and evaluate student trained from scratch.

distiller.fit(
    loader.load_training_data(),
    validation_data=loader.load_validation_data(),
    epochs=2,
    callbacks=[model_persistence.model_checkpoint_callback()]
)
student_scratch.evaluate(loader.load_validation_data())
