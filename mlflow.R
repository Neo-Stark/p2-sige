## -------------------------------------------------------------------------------------
## Sistemas Inteligentes para la Gestión en la Empresa
## Curso 2020-2021
## Juan Gómez Romero
## Adaptado al problema de fakenews por Francisco José González García
## -------------------------------------------------------------------------------------
library(reticulate)
use_condaenv('r-tensorflow')
library(keras)
if( ! ("mlflow" %in%  installed.packages()[,"Package"]) ) {
  install.packages("mlflow")
  library(mlflow)
  mlflow::install_mlflow()
}

library(mlflow)

# Lanzar script de entrenamiento
mlflow_run(entry_point = "clasificacion_binaria_mlflow.R")

# Visualizar en interfaz MLflow
# http://127.0.0.1:5987/
mlflow_ui()


# Cambiar valores de los parámetros

mlflow_run(entry_point = "clasificacion_binaria_mlflow.R", parameters = list(dropout_rate = 0.3, epochs = 10, hidden_units=128))

mlflow_run(entry_point = "clasificacion_binaria_mlflow.R", parameters = list(dropout_rate = 0.5, epochs = 15, batch_size=128))

mlflow_run(entry_point = "clasificacion_binaria_mlflow.R", parameters = list(dropout_rate = 0.3, epochs = 15, batch_size=50))


# Cargar modelo y realizar predicción
# logged_model = '/home/fran/MEGA/universidad/SIGE/p2/mlruns/0/f27672fa1a664cc79a043c9dad6c8c9f/artifacts/model'
# 
# loaded_model = mlflow_load_model(logged_model)
# mlflow_predict(loaded_model, '~/MEGA/universidad/SIGE/p2/datasets/mini50_twoClasses/train/1/1s9e0h.jpg')
