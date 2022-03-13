def train(model, train_generator, epochs, batch_size, test_generator, test_samples, train_samples):
    history = model.fit(
    train_generator,
    epochs=epochs,
    validation_data=test_generator,
    validation_steps= test_samples // batch_size,
    steps_per_epoch= train_samples // batch_size
    )
    model.save('../models/model.h5')
    return history