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
* [figuras](https://github.com/xavi-diaz/TFG/tree/main/figuras)
* [padel](https://github.com/xavi-diaz/TFG/tree/main/padel)
