import model_code.data_loader as data_loader
import model_code.model_architecture as model_architecture
import model_code.model_training as model_training
import model_code.model_analyzer as analyze_model
import model_code.model_persistence as model_persistence

if __name__ == '__main__':
    # load data
    data_loader = data_loader.DataLoader(img_height=40, img_width=40)
    train_ds = data_loader.load_training_data()
    val_ds = data_loader.load_validation_data()

    # create model
    model = model_architecture.model_architecture()

    # train model
    model_training.model_train(model, train_ds, val_ds)

    # save model
    model_persistence.model_save(model)

    # evaluate model
    analyze_model.model_evaluate(model)
