# Aplicaciones de Aprendizaje Máquina en Química. 
*Predicción de la longitud de onda de absorción máxima usando métodos de Machine Learning*

**Descripción de los archivos**
## Carpetas
* [CV/all features](https://github.com/xavi-diaz/TFG/tree/main/CV/all%20features) Contiene los datos divididos según andamio molecular usados para la validación cruzada.
* [Resultados](https://github.com/xavi-diaz/TFG/tree/main/CV/all%20features)
  - [CV](https://github.com/xavi-diaz/TFG/tree/main/Resultados/CV) Incluye los resultados de las validaciones cruzadas para las dos estrategias de separación de datos y los métodos k-NN y XGB.
  - [decision_tree](https://github.com/xavi-diaz/TFG/tree/main/Resultados/decision_tree) Incluye los resultados de los árboles de decisión.
  - [val_final](https://github.com/xavi-diaz/TFG/tree/main/Resultados/val_final) Incluye los resultados de las validaciones finales de los métodos k-NN y XGB.
* [chemprop](https://github.com/xavi-diaz/TFG/tree/main/chemprop)
  - [random_cv](https://github.com/xavi-diaz/TFG/tree/main/chemprop/random_cv) Incluye los resultados de la validación cruzada con división aleatoria para el método de Chemprop. 
  - [scaffold_cv](https://github.com/xavi-diaz/TFG/tree/main/chemprop/scaffold_cv) Incluye los resultados de la validación cruzada con división según andamio molecular para el método de Chemprop.
  - `preds.csv` contiene las predicciones obtenidas usando Chemprop en la validación final.
* [data_solvents](https://github.com/xavi-diaz/TFG/tree/main/data_solvents)
  - `chemfluor_cgsd_solvent_db.csv` Es la base de datos del descriptor molecular del disolvente Chemfluor CGSD
  - `mn_solvent_db.csv` Es la base de datos del descriptor molecular del disolvente Minnesota Solvent Descriptor
* [figuras](https://github.com/xavi-diaz/TFG/tree/main/figuras) Incluye las figuras.
* [padel](https://github.com/xavi-diaz/TFG/tree/main/padel) Software externo usado.
## Notebooks de Jupyter
* Tratamiento de datos, obtención de subconjuntos.ipynb
* Base de datos externa.ipynb
* `xgb_knn_distintas_features.ipynb` Incluye el código usado para evaluar los modelos de KNN y XGB usando las distintas representaciones del disolvente y con las dos divisiones de los datos estudiada. También incluye la validación final para KNN y XGB.
* `cv_chemprop.py` y `prediccion_chemprop.sh` incluyen el código que se ejecutó por terminal para llevar a cabo la validación cruzada de Chemprop y la validación final respectivamente.
* `k-NN_todas carcateristicas.ipynb` y `XGBoost_todas características.ipynb` incluyen unas pruebas iniciales donde se entrenaron KNN y XGB con todas las características.
## Archivos de datos
* `master_df_all_features.csv` Es la base de datos tomada de Greenman et al.
* `scaled_df.csv` Es la base de datos procesada y escaladas a media 0 y desviación estándar 1.
* `bval_final_all_features.csv` Es la base de datos usada para la validación final.
* `chemprop_data_val.csv` Incluye los datos indicados a Chemprop para llevar a cabo la validación final.
