# **Whale Re-ID: Identification & Embeddings Retrieval**

## Описание проекта

Система идентификации китов по фотографиям через **embedding-based retrieval**. Модель извлекает вектор признаков (embedding) из изображения, затем выполняется поиск ближайших соседей в базе эмбеддингов по **cosine similarity**.

---

## **Реализовано**

- **Извлечение эмбеддингов** и сравнение с базой по cosine similarity
- **Инференс/идентификация** (модуль WhaleIdentificationSystem)
- **Обучение модели** (ArcFace loss, оптимизация гиперпараметров)
- **Top-K retrieval** — поиск топ-100 ближайших китов из базы

### **Установка**

```

pip install -U torch torchvision numpy opencv-python pillow matplotlib transformers

```

**Как работает идентификация?**

A:
1. Извлекаем embedding изображения запроса (256-D вектор)
2. Вычисляем cosine similarity со всеми embeddings'ами в базе
3. Берем Top-100 индексов с наибольшей similarity
4. Возвращаем ID найденных китов

## **Параметры модели**

**DinoV2**
