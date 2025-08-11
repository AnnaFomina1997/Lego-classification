import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from PIL import Image
import json
import random

# Инициализация Flask-приложения
app = Flask(__name__)

# Загрузка модели
model = load_model('model.h5')

# Загрузка меток классов из JSON
with open('class_labels.json', 'r') as f:
    class_labels = json.load(f)

# Предобработка изображения для предсказания
def preprocess_image(image_path):
    img = Image.open(image_path).convert('L')  # Перевод в оттенки серого
    img = img.resize((512, 512))  # Изменение размера
    img_array = np.array(img) / 255.0  # Нормализация
    img_array = np.expand_dims(img_array, axis=-1)  # Добавление канала
    img_array = np.expand_dims(img_array, axis=0)  # Добавление размерности для батча
    return img_array

@app.route('/')
def index():
    return render_template('base.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join('static/uploads', filename)
            file.save(filepath)

            # Предобработка изображения
            img_array = preprocess_image(filepath)

            # Прогнозирование
            predictions = model.predict(img_array)
            predicted_class_idx = np.argmax(predictions, axis=1)[0]
            confidence = np.max(predictions) * 100
            predicted_class = class_labels[predicted_class_idx]  # Получаем метку класса

            # Округление уверенности до 3 знаков
            confidence = round(confidence, 3)

            # Поиск всех изображений предсказанного класса
            predicted_class_images = []
            all_images_folder = os.path.join('static', 'all_images')  # Путь к папке all_images

            for image_name in os.listdir(all_images_folder):
                if image_name.startswith(f"{predicted_class}-"):  # Изображения нужного класса
                    predicted_class_images.append(
                        url_for('static', filename=f"all_images/{image_name}")
                    )

            print("Изображения предсказанного класса:", predicted_class_images)  # Для отладки

            # Рендеринг HTML с изображениями
            return render_template(
                'predict.html',
                image_path=url_for('static', filename=f'uploads/{filename}'),
                predicted_class=predicted_class,
                confidence=confidence,
                predicted_class_images=predicted_class_images
            )
    return render_template('upload.html')



import random  # Для случайного выбора изображений

@app.route('/train', methods=['GET', 'POST'])
def train():
    global model  # Используем глобальную переменную модели
    if request.method == 'POST':
        folder = request.form.get('dataset_path')
        if not os.path.exists(folder):
            return redirect(request.url)

        # Загрузка данных для дообучения
        image_data = []
        labels = []
        new_classes = set()

        all_images_folder = os.path.join('static', 'all_images')
        os.makedirs(all_images_folder, exist_ok=True)  # Убедимся, что папка существует

        for filename in os.listdir(folder):
            if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                class_name = filename.split('-')[0]
                class_id = int(class_name)

                # Добавляем новый класс
                if class_id not in class_labels:
                    class_labels.append(class_id)
                    new_classes.add(class_id)

                index = class_labels.index(class_id)
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path).convert('L')
                img = img.resize((512, 512))
                img_array = np.array(img) / 255.0
                image_data.append(img_array)
                labels.append(index)

                # Добавление изображения в папку all_images
                target_path = os.path.join(all_images_folder, filename)
                if not os.path.exists(target_path):
                    original_img = Image.open(img_path)
                    original_img.save(target_path)

        # Сохранение обновленного списка классов
        with open('class_labels.json', 'w') as f:
            json.dump(class_labels, f)

        # Добавление старых изображений: 50% от общего числа новых данных
        old_image_files = [
            img for img in os.listdir(all_images_folder)
            if int(img.split('-')[0]) not in new_classes
        ]
        num_samples = int(15 * len(image_data))
        selected_old_images = random.sample(old_image_files, min(len(old_image_files), num_samples))

        for filename in selected_old_images:
            class_name = filename.split('-')[0]
            class_id = int(class_name)
            index = class_labels.index(class_id)

            img_path = os.path.join(all_images_folder, filename)
            img = Image.open(img_path).convert('L')
            img = img.resize((512, 512))
            img_array = np.array(img) / 255.0
            image_data.append(img_array)
            labels.append(index)

        # Преобразование данных
        image_data = np.expand_dims(np.array(image_data), axis=-1)
        labels = np.array(labels)
        num_classes = len(class_labels)
        labels = to_categorical(labels, num_classes=num_classes)

        # Обновление модели: пересоздание последнего слоя
        new_model = Sequential()
        for layer in model.layers[:-1]:  # Копируем все слои, кроме последнего
            layer.trainable = False  # Замораживаем слои
            new_model.add(layer)

        # Добавляем новый выходной слой
        new_model.add(Dense(num_classes, activation='softmax', name=f"output_dense_{num_classes}"))

        # Компиляция новой модели
        new_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Инициализация весов новой модели
        dummy_input = np.random.rand(1, 512, 512, 1)  # Размер входных данных модели
        new_model.predict(dummy_input)  # Инициализируем веса всех слоёв

        # Расширение весов последнего слоя
        old_weights, old_bias = model.layers[-1].get_weights()
        new_weights = np.random.uniform(
            low=-0.05, high=0.05, size=(old_weights.shape[0], num_classes)
        )
        new_weights[:, :old_weights.shape[1]] = old_weights

        new_bias = np.zeros((num_classes,))
        new_bias[:old_bias.shape[0]] = old_bias

        new_model.layers[-1].set_weights([new_weights, new_bias])

        # Замена глобальной модели
        model = new_model

        # Обучение без аугментации
        model.fit(image_data, labels, batch_size=32, epochs=30)

        # Оценка на полном наборе
        full_image_data = []
        full_labels = []

        for filename in os.listdir(all_images_folder):
            class_name = filename.split('-')[0]
            class_id = int(class_name)
            index = class_labels.index(class_id)
            img_path = os.path.join(all_images_folder, filename)
            img = Image.open(img_path).convert('L')
            img = img.resize((512, 512))
            img_array = np.array(img) / 255.0
            full_image_data.append(img_array)
            full_labels.append(index)

        full_image_data = np.expand_dims(np.array(full_image_data), axis=-1)
        full_labels = to_categorical(full_labels, num_classes=len(class_labels))

        full_loss, full_accuracy = model.evaluate(full_image_data, full_labels, verbose=0)

        # Сохранение модели
        model.save('model.h5')

        message = (
            f"Модель успешно дообучена!<br>"
            f"Точность на полном наборе: {full_accuracy:.2%}<br>"
            f"Потери на полном наборе: {full_loss:.4f}"
        )
        return render_template('train_result.html', message=message)

    return render_template('train.html')


if __name__ == '__main__':
    os.makedirs('static/uploads', exist_ok=True)
    app.run(debug=True)









