# Proyecto TFG - Modelos Probabilísticos y Explicabilidad en Deep Learning

Este repositorio contiene diferentes notebooks que exploran técnicas modernas de deep learning con enfoque en modelos probabilísticos, estimación de incertidumbre y explicabilidad, aplicados a tareas de regresión, clasificación y segmentación de imágenes.

## 📁 Contenidos

### `Datos-Segmentacion.ipynb`
Preprocesamiento y preparación de datos para el proyecto de segmentación de imágenes. Aquí se organizan los datos de entrada que se utilizan en modelos posteriores.

### `Segmentacion-Imagenes.ipynb`
Comparativa entre modelos deterministas y bayesianos (MC Dropout) sobre la misma arquitectura de SegNet en tareas de **segmentación de imágenes**. Además, se incorpora:
- 🎯 Evaluación de la incertidumbre en las predicciones
- 🔁 Skip connections
- 🔍 Técnicas de explicabilidad como **Grad-CAM**

### `Sistema-Frenado.ipynb`
Aplicación práctica del modelo con MC Dropout en un sistema de frenado. Se analiza el comportamiento del modelo en un escenario de decisión real, donde la estimación de incertidumbre es crítica.

### `IncertidumbreRegresion.ipynb`
Ejemplo didáctico de regresión donde se compara:
- Modelos deterministas
- Modelos de inferencia variacional (VI) usando **TensorFlow Probability (TFP)**
- Métodos bayesianos avanzados como **NUTS** usando **TensorFlow Probability (TFP)**

Se ilustra cómo varía la incertidumbre según el enfoque adoptado.

### `bbb-clasificacion-f.ipynb`
Implementación de una red neuronal convolucional bayesiana usando **Bayes By Backprop (BBB)**, inspirada en la implementación de **Shrimar**. Se aplica al dataset **CIFAR-10** para tareas de clasificación, comparando rendimiento y comportamiento frente a redes deterministas.

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
Explorar y aplicar técnicas de deep learning bayesiano y de explicabilidad para evaluar modelos más robustos, confiables y transparentes. El enfoque se centra en:
- Entender cómo y por qué los modelos hacen sus predicciones
- Medir la incertidumbre asociada a dichas predicciones
- Aplicar estas ideas a problemas reales de clasificación, regresión y segmentación

---

## 👤 Autor
**Fernando Francisco Moya Rangel**  
Trabajo de Fin de Grado (TFG)
