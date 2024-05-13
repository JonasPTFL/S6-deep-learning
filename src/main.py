import src.model_code.mode_iteration as mode_iteration
import tensorflow as tf

if __name__ == '__main__':
    # Print the gpus available (test for local development)
    print("GPUs available: ", tf.config.list_physical_devices('GPU'))

    # create model iteration
    model_iteration = mode_iteration.ModelIteration(
        iteration_name='test1',
    )
    # run model iteration
    model_iteration.run()
