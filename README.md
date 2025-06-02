# TFG - Métodos de Inferencia en Redes Neuronales Bayesianas

Este repositorio contiene diferentes notebooks que exploran técnicas modernas de deep learning con enfoque en modelos probabilísticos, estimación de incertidumbre y explicabilidad, aplicados a tareas de regresión, clasificación y segmentación de imágenes.

## 📁 Contenidos

### `Datos-Segmentacion.ipynb`
Preprocesamiento y preparación de datos de **CityScapes** para el proyecto de segmentación de imágenes. Aquí se organizan y procesan los datos de entrada que se utilizan en modelos posteriores.

### `Segmentacion-Imagenes.ipynb`
Comparativa entre modelos deterministas y bayesianos (MC Dropout) sobre la misma arquitectura de SegNet en tareas de **segmentación de imágenes**. Además, se incorpora:
- 🎯 Evaluación de la incertidumbre en las predicciones
- 🔁 Skip connections
- 🔍 Técnicas de explicabilidad como **Grad-CAM**

### `Sistema-Frenado.ipynb`
Aplicación práctica del modelo con MC Dropout en un sistema de frenado. Se analiza la calidad de las segmentaciones de los modelos anteriores en un escenario de decisión real, donde la estimación de incertidumbre es crítica.

### `IncertidumbreRegresion.ipynb`
Ejemplo didáctico de regresión donde se compara:
- Red neuronal determinista
- Red neuronal bayesiana con inferencia variacional (VI) usando **TensorFlow Probability (TFP)**
- Red neuronal bayesiana con MCMC, en concreto **NUTS** usando **TensorFlow Probability (TFP)**

Se muestra cómo varía la incertidumbre según el enfoque adoptado.

### `bbb-clasificacion-f.ipynb`
Implementación de una red neuronal convolucional bayesiana usando **Bayes By Backprop (BBB)** con el local reparametrization trick. Se aplica al dataset **CIFAR-10** para tareas de clasificación, comparando rendimiento y comportamiento frente a redes deterministas.

---

## 🧠 Tecnologías utilizadas
- TensorFlow & TensorFlow Probability
- PyTorch
- Grad-CAM
- MC Dropout
- Inferencia Variacional (VI)
- NUTS (No-U-Turn Sampler)
- CIFAR-10

---

## 📌 Objetivo
Explorar y aplicar técnicas de deep learning bayesiano para evaluar modelos más robustos, confiables y transparentes. El enfoque se centra en:
- Entender las principales ventajas y desventajas de las redes neuronales bayesianas
- Medir la incertidumbre asociada a las predicciones
- Aplicar estas ideas a problemas reales de clasificación, regresión y segmentación
- Sacar conclusiones de los métodos de inferencia bayesiana más óptimos en la práctica

---
## 🛠️ Recursos utilizados
El desarrollo del Trabajo de Fin de Grado se ha llevado a cabo principalmente utilizando los siguientes entornos y plataformas:

- **Google Colab Pro**: entorno principal para el entrenamiento y evaluación de los modelos en la segmentación de imágenes. Se ha aprovechado el acceso a GPU (Tesla A100) para acelerar el entrenamiento de las redes convolucionales con tantas capas y parámetros a optimizar.
- **Kaggle**: empleado en los experimentos de regresión y clasificación. Gracias al acceso gratuito a GPU y al entorno reproducible de notebooks, fue posible acelerar el proceso de entrenamiento y almacenar los resultados de forma eficiente.

Estas herramientas han permitido un flujo de trabajo flexible y eficiente en la experimentación con modelos bayesianos de deep learning.

---


## 👤 Autor
**Fernando Francisco Moya Rangel**  
Trabajo de Fin de Grado (TFG)

---

## 📚 Referencias

- Shridhar, K., Laumann, F., & Liwicki, M. (2019). *A comprehensive guide to Bayesian convolutional neural network with variational inference*. arXiv preprint [arXiv:1901.02731](https://arxiv.org/abs/1901.02731)

- Shridhar, K., Laumann, F., & Liwicki, M. (2018). *Uncertainty estimations by softplus normalization in Bayesian convolutional neural networks with variational inference*. arXiv preprint [arXiv:1806.05978](https://arxiv.org/abs/1806.05978)
  
- Kendall, A., Badrinarayanan, V., & Cipolla, R. (2016). *Bayesian SegNet: Model Uncertainty in Deep Convolutional Encoder-Decoder Architectures for Scene Understanding*. arXiv preprint [arXiv:1511.02680](https://arxiv.org/abs/1511.02680)
  
- Duerr, O., Loui, A., & Luckow, C. (2021). *Probabilistic Deep Learning with Python, Keras and TensorFlow Probability*. Addison-Wesley Professional.

