# ml_business_course_progect
python-flask-docker

Итоговый проект курса "Машинное обучение в бизнесе"

Стек:

ML: sklearn, pandas, numpy,  keras. API: flask Данные: CIFAR-10 dataset

Задача: предсказать по фото класс изображения. Всего 10 классов: airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships,  trucks. 

Состав датасета:
60000 цветных изображений разрешением 32x32
    

Модель: AlexNet CNN
Клонируем репозиторий и создаем образ

$ git clone https://github.com/mr-rider/ml_business_course_progect
$ cd ml_business_course_progect
$ docker build -t ml_business_course_progect .

Запускаем контейнер

Здесь Вам нужно создать каталог локально и сохранить туда предобученную модель (<your_local_path_to_pretrained_models> нужно заменить на полный путь к этому каталогу)

$ docker run -d -p 5000:5000 -p -v <your_local_path_to_pretrained_models>:/app/app/models ml_business_course_progect

Пример запроса:
curl -X POST -F image=@car.jpg 'http://localhost:5000/predict'

