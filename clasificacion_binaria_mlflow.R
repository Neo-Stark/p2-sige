## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2020-2021
## Juan Gómez Romero
## Adaptado al problema de fakenews por Francisco José González García
## -------------------------------------------------------------------------------------

library(reticulate)
use_condaenv('r-tensorflow')
library(keras)

#### WORKARROUND para que TF funcione en GPU, comentar las lineas si se actualiza
# y se resuelve el bug https://github.com/tensorflow/tensorflow/issues/43174
library(tensorflow)
physical_devices <- tf$config$list_physical_devices('GPU')
tf$config$experimental$set_memory_growth(physical_devices[[1]], TRUE)
####
library(mlflow)


hidden_units      <- mlflow_param("hidden_units", 100, "integer", "Number of units of the hidden layer")
hidden_activation <- mlflow_param("hidden_activation", "relu", "string", "Activation function for the hidden layer")
dropout_rate      <- mlflow_param("dropout_rate", 0.5, "numeric", "Dropout rate (after the hidden layer)")
epsilon           <- mlflow_param("epsilon", 0.01, "numeric", "Epsilon parameter of the batch normalization (after convolution)")
batch_size        <- mlflow_param("batch_size", 64, "integer", "Mini-batch size")
epochs            <- mlflow_param("epochs", 5, "integer", "Number of training epochs")

## -------------------------------------------------------------------------------------
## Cargar y pre-procesar datos

# Directorios
dataset_dir           <- './datasets/medium10000_twoClasses/'

train_images_dir      <- paste0(dataset_dir, 'train')
val_images_dir        <- paste0(dataset_dir, 'val')
test_images_dir       <- paste0(dataset_dir, 'test')

# Generadores
train_images_generator <- image_data_generator(rescale = 1/255)
val_images_generator   <- image_data_generator(rescale = 1/255)
test_images_generator  <- image_data_generator(rescale = 1/255)

# Flujos
train_generator_flow <- flow_images_from_directory(
  directory = train_images_dir,
  generator = train_images_generator,
  class_mode = 'categorical',
  batch_size = batch_size,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

validation_generator_flow <- flow_images_from_directory(
  directory = val_images_dir,
  generator = val_images_generator,
  class_mode = 'categorical',
  batch_size = batch_size,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

test_generator_flow <- flow_images_from_directory(
  directory = test_images_dir,
  generator = test_images_generator,
  class_mode = 'categorical',
  batch_size = batch_size,
  target_size = c(64, 64)         # (w x h) --> (64 x 64)
)

## -------------------------------------------------------------------------------------
## Crear modelo

# Definir arquitectura
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 128,  kernel_size = c(3, 3), activation = hidden_activation, input_shape = c(64, 64, 3)) %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128,  kernel_size = c(3, 3), activation = hidden_activation) %>% layer_batch_normalization(epsilon = epsilon) %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = hidden_activation) %>% layer_batch_normalization(epsilon = epsilon) %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = hidden_activation) %>% layer_batch_normalization(epsilon = epsilon) %>% layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = hidden_units, activation = hidden_activation) %>%
  layer_dropout(rate = dropout_rate) %>% 
  layer_dense(units = 2, activation = "softmax")

summary(model)

# Compilar modelo
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(),
  metrics = c('accuracy')
)

## -------------------------------------------------------------------------------------
## MLflow
with(mlflow_start_run(), {

  # Entrenar modelo
  history <- model %>% 
    fit(
      train_generator_flow, 
      validation_data = validation_generator_flow,
      steps_per_epoch = train_generator_flow$samples / batch_size,
      epochs = epochs
    )
  
  # Visualizar entrenamiento
  plot(history)
  
  # Calcular metricas sobre datos de validación
  metrics <- model %>% 
    evaluate_generator(test_generator_flow, steps = 1)
  
  # Guardar valores interesantes de la ejecución
  mlflow_log_param("dropout_rate", dropout_rate)
  mlflow_log_param("epochs", epochs)
  mlflow_log_param("batch_size", batch_size)
  
  mlflow_log_metric("loss", metrics["loss"])
  mlflow_log_metric("accuracy", metrics["accuracy"])
  
  # Guardar modelo
  mlflow_log_model(model, "model")
  
  # Mostrar salida
  message("CNN model (dropout_rate=", dropout_rate, ", epochs=", epochs, ", batch_size= ", batch_size, "):")
  message("  loss: ", metrics["loss"])
  message("  accuracy: ", metrics["accuracy"])
})
