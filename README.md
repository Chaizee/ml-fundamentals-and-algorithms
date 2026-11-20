# Лабораторный практикум по машинному обучению (OMSTU ML Labs)

Репозиторий представляет собой единый структурированный комплекс лабораторных работ по курсу **«Машинное обучение и большие данные»**, охватывающий все ключевые разделы классического ML и глубокого обучения: от разведочного анализа данных (EDA), линейных моделей, метрических/байесовских/ядерных классификаторов и ансамблей решающих деревьев до кластеризации, снижения размерности, полносвязных нейронных сетей (MLP с обратным распространением ошибки) и свёрточных архитектур (CNN / Transfer Learning).

Все лабораторные работы сведены в **единый пул профессионально оформленных Jupyter-ноутбуков (`.ipynb`)** в корне репозитория с теоретическими пояснениями, воспроизводимым кодом, вшитыми графиками и аналитическими выводами.

---

## Структура репозитория

```text
OMSTU_ML_LABS/
├── lab1_eda_classification.ipynb      # ЛР №1.1: EDA и предобработка для задачи классификации (Diabetes)
├── lab1_eda_regression.ipynb          # ЛР №1.2: EDA и предобработка для задачи регрессии (Wine Quality)
├── lab2_linear_regression.ipynb       # ЛР №2: Линейная регрессия, L1/L2/ElasticNet, полиномы и свой SGD
├── lab3_classification.ipynb          # ЛР №3: Классификация (LogReg, KNN, Naive Bayes, SVM), SMOTE и свой KNN
├── lab4_trees_ensembles.ipynb         # ЛР №4: Решающие деревья, Bagging, Boosting (XGB/LGBM/CatBoost), Stacking
├── lab5_clustering.ipynb              # ЛР №5: Кластеризация (KMeans, DBSCAN, GMM, Ward, AP) и свой KMeans
├── lab6_dimensionality_reduction.ipynb# ЛР №6: Отбор признаков (VT, SelectKBest, RFE) и снижение размерности (PCA, t-SNE, Isomap, UMAP)
├── lab7_mlp_neural_networks.ipynb     # ЛР №7: Многослойный перцептрон (MLP) и Backpropagation с нуля (XOR + табличные данные)
├── lab8_cnn_transfer_learning.ipynb   # ЛР №8: Свёрточные нейросети (Custom CNN) и Fine-Tuning (VGG19, InceptionV3)
├── csv/                               # Табличные наборы данных (исходные и очищенные)
│   ├── cl.csv                         # Исходный датасет CDC Diabetes Health Indicators (253 680 строк)
│   ├── cl_finished.csv                # Очищенный датасет для классификации (229 781 строка, 22 столбца)
│   ├── winequality-red.csv            # Исходный датасет красного вина (1 599 строк)
│   ├── winequality-white.csv          # Исходный датасет белого вина (4 898 строк)
│   └── winequality_finished.csv       # Объединённый датасет для регрессии (6 497 строк, 13 столбцов)
├── data/                              # Наборы изображений для компьютерного зрения (ЛР №8)
│   ├── data3/                         # Cats vs Dogs (2 000 изображений: cats, dogs)
│   └── 101_ObjectCategories/          # Caltech-101 subset (233 изображения: brain, pizza, revolver)
└── README.md                          # Описание проекта, структуры, методологии и результатов
```

---

## Используемые наборы данных

1. **Задача табличной классификации (`csv/cl.csv` → `csv/cl_finished.csv`):**
   * **Источник:** *CDC Behavioral Risk Factor Surveillance System (BRFSS 2015)*.
   * **Целевой признак:** `Diabetes_012` — 3 класса (`0` — диабет отсутствует, `1` — преддиабет, `2` — сахарный диабет).
   * **Признаки:** 21 кардиометаболический, поведенческий и социально-демографический показатель (`HighBP`, `HighChol`, `BMI`, `GenHlth`, `PhysHlth`, `MentHlth`, `DiffWalk`, `Age`, `Income` и др.).
   * **Объём:** 253 680 исходных анкет → **229 781** уникальная запись после удаления 23 899 дубликатов.

2. **Задача табличной регрессии (`csv/winequality-red.csv` + `csv/winequality-white.csv` → `csv/winequality_finished.csv`):**
   * **Источник:** *UCI Wine Quality Dataset (Vinho Verde, Portugal)*.
   * **Целевой признак:** `quality` — экспертная оценка качества вина (от 3 до 9 баллов).
   * **Признаки:** 11 физико-химических свойств (`fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`, `free/total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`) + бинарный индикатор типа вина `red` (`1` — красное, `0` — белое).
   * **Объём:** **6 497** образцов (1 599 красного и 4 898 белого вина; присутствуют 1 177 полных дубликатов строк).

3. **Задачи компьютерного зрения (`data/`):**
   * **`data/data3` (Cats vs Dogs):** 2 000 RGB-изображений кошек и собак (1 600 train / 400 validation) для обучения собственной свёрточной сети.
   * **`data/101_ObjectCategories` (Caltech-101 subset):** 233 изображения по 3 выбранным классам (`brain`, `pizza`, `revolver`: 188 train / 45 validation) из 101 категории датасета `Caltech-101` для исследования Transfer Learning и Fine-Tuning.

---

## Подробное описание ноутбуков и полученных результатов

### 1. [`lab1_eda_classification.ipynb`](lab1_eda_classification.ipynb) — Разведочный анализ данных (Классификация)
* **Суть работы:** первичная диагностика структуры данных BRFSS, проверка целочисленности признаков, фильтрация дубликатов, расчёт ранговой корреляции Кендалла $\tau$, анализ распределений (`histplot`, `violinplot`, `boxplot`) и баланса классов.
* **Что получилось в итоге:**
  * Пропуски отсутствуют; удалено **23 899 дубликатов** (9.42 % выборки), сохранено **229 781** наблюдение в `csv/cl_finished.csv`.
  * Выявлен сильный дисбаланс классов (после очистки): `0` — **82.71 %** (190 055), `2` — **15.27 %** (35 097), `1` — **2.01 %** (4 629), что обосновало применение стратификации и оверсемплинга `SMOTE` в последующих работах.
  * Определены наиболее информативные факторы риска диабета по ранговой корреляции Кендалла: общая оценка здоровья `GenHlth` ($\tau = 0.28$), артериальное давление `HighBP` ($\tau = 0.26$), трудности при ходьбе `DiffWalk` ($\tau = 0.21$), холестерин `HighChol` ($\tau = 0.20$), индекс массы тела `BMI` ($\tau = 0.19$) и возраст `Age` ($\tau = 0.18$).

---

### 2. [`lab1_eda_regression.ipynb`](lab1_eda_regression.ipynb) — Разведочный анализ данных (Регрессия)
* **Суть работы:** объединение выборок красного и белого вина с введением признака `red`, исследование распределений физико-химических свойств и матриц корреляции Пирсона, построение сравнительных диаграмм размаха (`boxplot`) в разрезе балла качества `quality`.
* **Что получилось в итоге:**
  * Сформирован единый датасет из **6 497 образцов** и 12 признаков (`csv/winequality_finished.csv`, в том числе 1 177 повторяющихся строк).
  * Установлено, что **76.62 %** оценок сосредоточены на значениях `5` (32.91 %, 2 138 шт.) и `6` (43.65 %, 2 836 шт.) баллов.
  * Главным положительным фактором качества выступает содержание спирта `alcohol` ($r = +0.4443$), а отрицательными — плотность `density` ($r = -0.3059$), летучая кислотность `volatile acidity` ($r = -0.2657$) и хлориды `chlorides` ($r = -0.2007$).
  * Выявлена сильная мультиколлинеарность между `free sulfur dioxide` и `total sulfur dioxide` ($r = +0.7209$), `density` и `alcohol` ($r = -0.6867$), `density` и `residual sugar` ($r = +0.5525$), требующая регуляризации в линейных моделях.

---

### 3. [`lab2_linear_regression.ipynb`](lab2_linear_regression.ipynb) — Линейные модели регрессии и регуляризация
* **Суть работы:** построение линейных моделей регрессии (МНК, `Lasso` $L_1$, `Ridge` $L_2$, `ElasticNet`, полиномиальная регрессия), подбор гиперпараметров на кросс-валидации тремя способами (`Optuna`, `GridSearchCV`, `RandomizedSearchCV`), ручная реализация всех метрик регрессии ($MSE, RMSE, MAE, MAPE, R^2$) и собственная реализация линейного регрессора с $L_2$-регуляризацией на мини-батчевом стохастическом градиентном спуске (`LinearRegressionSGD`).
* **Что получилось в итоге (на тестовой выборке):**

| Модель | Метод подбора | MSE | RMSE | MAE | MAPE | $R^2$ |
|---|---|:---:|:---:|:---:|:---:|:---:|
| **PolynomialFeatures (deg=2) + Ridge** | Optuna (CV) | **0.5051** | **0.7107** | **0.5575** | **0.0998** | **0.3423** |
| **LinearRegression (МНК)** | Optuna (GridSampler) | 0.5378 | 0.7333 | 0.5710 | 0.1023 | 0.2997 |
| **Lasso ($L_1$)** | Optuna (CV) | 0.5379 | 0.7334 | 0.5711 | 0.1023 | 0.2996 |
| **LinearRegressionSGD (своя, $L_2$)** | Ручная реализация | 0.5379 | 0.7334 | 0.5713 | 0.1024 | 0.2996 |
| **ElasticNet ($L_1 + L_2$)** | RandomizedSearchCV | 0.5379 | 0.7334 | 0.5711 | 0.1023 | 0.2996 |
| **Ridge ($L_2$)** | GridSearchCV | 0.5379 | 0.7334 | 0.5711 | 0.1023 | 0.2995 |

  * Добавление полиномиальных признаков 2-й степени с $L_2$-регуляризацией увеличило $R^2$ с `0.2997` до **`0.3423`**, тогда как полином 3-й степени без регуляризации переобучается ($MSE_{test} = 0.6275 > MSE_{train} = 0.4272$).
  * Веса собственного класса `LinearRegressionSGD` совпали с аналитическим решением `Ridge` с точностью до $10^{-3}$, а ручные метрики — на 100 % с `sklearn.metrics`.

---

### 4. [`lab3_classification.ipynb`](lab3_classification.ipynb) — Методы классификации и борьба с дисбалансом классов
* **Суть работы:** решение 3-классовой задачи диагностики диабета с помощью `LogisticRegression`, `KNeighborsClassifier`, `GaussianNB` и `SVC`; устранение дисбаланса классов методом `SMOTE` в едином конвейере со стандартизацией `StandardScaler`; подбор гиперпараметров через `Optuna`; ручная реализация матрицы ошибок, метрик `Precision`, `Recall`, `F1`, `Accuracy`, многоклассового `ROC-AUC` (через ранговую статистику Манна–Уитни) и собственного векторного классификатора `MyKNN`.
* **Что получилось в итоге (на стратифицированной тестовой подвыборке 3 000 объектов):**
  * **Gaussian Naive Bayes (`var_smoothing = 0.0566`):** лидер по `Accuracy = 0.6927`, `Precision (weighted) = 0.8087`, `F1 (weighted) = 0.7365`, `F1 (macro) = 0.4366`, `ROC-AUC (OvR) = 0.7429`.
  * **Logistic Regression (`C = 0.0014`, `l1`, `saga`):** лидер по качеству вероятностного ранжирования `ROC-AUC (OvR) = 0.7689` (`Accuracy = 0.6303`, `Precision (weighted) = 0.8194`, `F1 (weighted) = 0.6951`).
  * **SVM (`C = 9.09`, `gamma = 0.0255`, `rbf`):** `Accuracy = 0.6593`, `F1 (weighted) = 0.7130`, `ROC-AUC (OvR) = 0.7289`.
  * **KNN (`n_neighbors = 17`, `manhattan`, `distance`):** `Accuracy = 0.6423`, `F1 (weighted) = 0.6974`, `ROC-AUC (OvR) = 0.6808`.
  * Собственная реализация `MyKNN` (с поддержкой весов `uniform` и `distance`) совпала с `sklearn.neighbors.KNeighborsClassifier` на **99.53 %** тестовых объектов при равномерных весах (`Accuracy = 0.6317` vs `0.6337`) и на **100 %** при взвешивании по обратному расстоянию (`Accuracy = 0.6423`), а ручные метрики полностью совпали с эталонными функциями `scikit-learn`.

---

### 5. [`lab4_trees_ensembles.ipynb`](lab4_trees_ensembles.ipynb) — Решающие деревья и ансамбли моделей
* **Суть работы:** комплексное исследование и визуализация деревьев решений (`CART`), классических ансамблей (`Bagging`, `GradientBoosting`, `Stacking`) и современных библиотек градиентного бустинга (`XGBoost`, `LightGBM`, `CatBoost`) с оптимизацией гиперпараметров в `Optuna`, а также написание с нуля трёх собственных алгоритмов: `CustomDecisionTree` (Classifier & Regressor), `CustomBagging` и Out-Of-Fold `CustomStacking`.
* **Что получилось в итоге (сравнение 20 обученных моделей):**
  * **Регрессия (`Wine Quality`):**
    * Одиночные деревья: `DecisionTreeRegressor` ($R^2 = 0.2735$, $MSE = 0.5301$, $MAE = 0.5621$), `CustomDecisionTreeRegressor` ($R^2 = 0.2634$, $MSE = 0.5375$, $MAE = 0.5736$).
    * Бэггинг и стекинг: `BaggingRegressor` ($R^2 = 0.3597$, $MSE = 0.4672$), `CustomBaggingRegressor` ($R^2 = 0.4080$, $MSE = 0.4320$), `CustomStackingRegressor` ($R^2 = 0.4112$, $MSE = 0.4296$), `StackingRegressor` ($R^2 = 0.4513$, $MSE = 0.4004$).
    * Градиентный бустинг (лидеры): **`GradientBoostingRegressor`** ($R^2 = \mathbf{0.5083}$, $MSE = \mathbf{0.3588}$, $MAE = \mathbf{0.4400}$), **`CatBoostRegressor`** ($R^2 = \mathbf{0.5042}$, $MSE = 0.3617$, $MAE = 0.4651$), **`XGBRegressor`** ($R^2 = \mathbf{0.4937}$, $MSE = 0.3694$, $MAE = 0.4535$), `LGBMRegressor` ($R^2 = 0.4552$, $MSE = 0.3975$).
  * **Классификация (`Diabetes_012`, тестовая выборка при обучении на SMOTE-выборке):**
    * Лучшие показатели взвешенной $F_1$-меры и `Recall` показали модели с умеренной глубиной, устойчивые к переобучению на синтетических SMOTE-примерах: `CustomDecisionTreeClassifier` ($F_1 = \mathbf{0.7204}$, $Recall = \mathbf{0.6800}$), `CustomStackingClassifier` ($F_1 = \mathbf{0.7135}$, $Recall = 0.6693$), `CustomBaggingClassifier` ($F_1 = \mathbf{0.6928}$, $Recall = 0.6290$) и `StackingClassifier` ($F_1 = 0.6795$, $Precision = 0.8029$), за которыми следуют `BaggingClassifier` ($F_1 = 0.6775$, $Precision = \mathbf{0.8220}$), `CatBoostClassifier` ($F_1 = 0.6764$), `XGBClassifier` ($F_1 = 0.6726$), `GradientBoostingClassifier` ($F_1 = 0.6610$) и `LGBMClassifier` ($F_1 = 0.6602$).

---

### 6. [`lab5_clustering.ipynb`](lab5_clustering.ipynb) — Кластеризация данных
* **Суть работы:** исследование 6 алгоритмов кластеризации (`KMeans`, `DBSCAN`, `GaussianMixture`, `AgglomerativeClustering` со связью Уорда, `AffinityPropagation` и собственной реализации `MyKMeans`) на трёх датасетах: `clf` (`make_classification`, 3 кластера), `blob` (`make_blobs`, 4 кластера) и `real` (`Diabetes_012`). Подбор числа кластеров и параметров по методу локтя, коэффициенту силуэта, графику $k$-расстояний и дендрограммам.
* **Что получилось в итоге:**
  * На компактных гауссовых сгустках (`blob`) `AgglomerativeClustering` и `MyKMeans` достигают **`ARI = 0.9991`**, **`NMI = 0.9982`**, **`Silhouette = 0.7975`**, **`Davies-Bouldin = 0.2833`**, а `KMeans`, `GaussianMixture` и `AffinityPropagation` — `ARI = 0.9982`, `NMI = 0.9963`, `Silhouette = 0.7975`, `Davies-Bouldin = 0.2833` (`DBSCAN`: `ARI = 0.9951`, `NMI = 0.9893`).
  * На вытянутых перекрывающихся кластерах (`clf`) лидирует вероятностная модель **`GaussianMixture`** (**`ARI = 0.7862`**, **`NMI = 0.7503`**, `Silhouette = 0.4632`), опережая `AgglomerativeClustering` (`ARI = 0.7311`, `NMI = 0.7162`), `KMeans` (`ARI = 0.6978`, `NMI = 0.6558`, `Silhouette = 0.4769`) и `MyKMeans` (`ARI = 0.6927`, `NMI = 0.6498`, `Silhouette = 0.4764`).
  * На реальных 21-мерных медицинских данных (`real`, 3 000 объектов) `KMeans` и `MyKMeans` выделяют 2 макрокластера состояния здоровья с практически идентичными метриками (`ARI = 0.1854` и `0.1853`, `NMI = 0.0804` и `0.0807`, `Silhouette = 0.1776` и `0.1774`, `Davies-Bouldin = 2.6070` и `2.6096`).

---

### 7. [`lab6_dimensionality_reduction.ipynb`](lab6_dimensionality_reduction.ipynb) — Отбор признаков и снижение размерности
* **Суть работы:** сравнение методов фильтрации и обёрточного отбора признаков (`VarianceThreshold`, `SelectKBest`, `RFE`) с линейными и нелинейными методами снижения размерности (`PCA`, `KernelPCA` с ядрами `poly/rbf/sigmoid`, `t-SNE`, `Isomap`, `UMAP`) на задачах классификации и регрессии, а также собственная реализация метода главных компонент `CustomPCA` через спектральное разложение ковариационной матрицы с визуализацией кластеров `MyKMeans`.
* **Что получилось в итоге:**
  * В задаче регрессии (`Wine Quality`, базовое качество на 12 признаках: $R^2 = 0.4302$, $MAE = 0.4607$) методы выделения главных компонент и фильтрации по дисперсии **улучшили качество ансамбля** за счёт декорреляции коллинеарных переменных: **`KernelPCA (sigmoid, 5)`** достиг $R^2 = \mathbf{0.4607}$, $MAE = \mathbf{0.4586}$, **`PCA (7)`** — $R^2 = \mathbf{0.4479}$, $MAE = \mathbf{0.4584}$, а **`VarianceThreshold (10)`** — $R^2 = \mathbf{0.4322}$, $MAE = 0.4613$ (тогда как отбор 7 исходных признаков методом `RFE` дал $R^2 = 0.4214$, $MAE = 0.4695$).
  * В задаче классификации (`Diabetes_012`, базовое качество на 21 признаке: `Accuracy = 0.8083`, `F1 (weighted) = 0.7792`, `F1 (macro) = 0.3881`) сжатие с 21 до 12 признаков улучшило целевые метрики: **`PCA (12)`** повысил точность до **`Accuracy = 0.8139`** (`F1-w = 0.7796`), **`VarianceThreshold (19)`** — до **`F1-w = 0.7805`** (`Accuracy = 0.8083`), а **`RFE (k=12)`** достиг лучшего **`F1 (macro) = 0.3983`** (`Accuracy = 0.8032`, `F1-w = 0.7797`).
  * Собственная реализация `CustomPCA` полностью совпала с `sklearn.decomposition.PCA` и позволила отобразить структуру кластеров `MyKMeans` на плоскости первых двух главных компонент (объясняющих `16.84 %` и `8.35 %` дисперсии).

---

### 8. [`lab7_mlp_neural_networks.ipynb`](lab7_mlp_neural_networks.ipynb) — Многослойный перцептрон (MLP) и Backpropagation
* **Суть работы:** математический вывод и полная реализация с нуля на `NumPy` двухслойного перцептрона (`Perceptron` и `MLPRegressorScratch`) с прямым проходом и алгоритмом обратного распространения ошибки (*Backpropagation*), демонстрация на нелинейной задаче `XOR`, а также сравнение с `MLPClassifier` и `MLPRegressor` из `scikit-learn` на реальных датасетах курса.
* **Что получилось в итоге:**
  * На задаче **XOR** собственный перцептрон со скрытым слоем из 4 нейронов сходится со значения `MSE = 0.25568` до **`MSE = 0.00035`** (на 8 000-й эпохе) и со **100 %** точностью разделяет нелинейные классы (построена 2D карта разделяющей поверхности).
  * На сбалансированной 3-классовой задаче **`Diabetes_012`** собственный `Perceptron (NumPy, своя)` показал `Accuracy = 0.4907`, `F1 (macro) = 0.4869`, `ROC-AUC (OvR) = 0.6795`, практически совпав с `MLPClassifier (sklearn)` (`Accuracy = 0.4924`, `F1 (macro) = 0.4867`, `ROC-AUC (OvR) = 0.6878`).
  * На задаче регрессии **`Wine Quality`** собственный `MLPRegressorScratch (NumPy, своя)` достиг $R^2 = \mathbf{0.3134}$ ($MSE = 0.5102$, $MAE = 0.5562$), а двухслойный `MLPRegressor (sklearn, 64-32 ReLU)` — $R^2 = \mathbf{0.3495}$ ($MSE = 0.4834$, $MAE = 0.5357$), превзойдя все линейные и полиномиальные модели из ЛР №2.

---

### 9. [`lab8_cnn_transfer_learning.ipynb`](lab8_cnn_transfer_learning.ipynb) — Свёрточные нейронные сети и Transfer Learning
* **Суть работы:** построение и обучение с нуля собственной 3-блочной свёрточной сети (`Conv2D + MaxPooling2D + Flatten + Dense + Dropout`) с аугментацией изображений на датасете `Cats vs Dogs` (`data/data3`), а также перенос обучения (*Transfer Learning*) и тонкая настройка (*Fine-Tuning*) верхних слоёв предобученных на `ImageNet` архитектур **VGG19** (`block5`) и **InceptionV3** (`mixed9+`) на 3 классах набора `Caltech-101` (`data/101_ObjectCategories`).
* **Что получилось в итоге:**

| Архитектура | Набор данных | Всего параметров | Обучаемых параметров | Эпох | Train Accuracy | Val Loss | Val Accuracy |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Custom CNN (обучение с нуля)** | Cats vs Dogs (2 000 изобр.) | 9,494,081 | 9,494,081 | 15 | 71.06 % | 0.5979 | **66.25 %** |
| **VGG19 (Fine-Tuning `block5`)** | Caltech-101 (3 класса, 233 изобр.) | 20,156,483 | 9,571,331 | 10 | 93.09 % | **0.0696** | **100.00 %** |
| **InceptionV3 (Fine-Tuning `mixed9+`)** | Caltech-101 (3 класса, 233 изобр.) | 22,328,099 | 6,598,851 | 10 | 85.11 % | 0.1846 | **97.78 %** |

---

## Алгоритмы и метрики, реализованные с нуля (From Scratch)

| Ноутбук | Класс / Функция | Описание собственной реализации |
|---|---|---|
| [`lab2_linear_regression.ipynb`](lab2_linear_regression.ipynb) | `manual_metrics`, `LinearRegressionSGD` | Ручной расчёт `MSE`, `RMSE`, `MAE`, `MAPE`, $R^2$; линейная регрессия с $L_2$-штрафом на мини-батчевом SGD |
| [`lab3_classification.ipynb`](lab3_classification.ipynb) | `manual_confusion_matrix`, `manual_metrics`, `MyKNN` | Матрица ошибок, `Precision`, `Recall`, $F_1$, многоклассовый `ROC-AUC` (статистика Манна–Уитни); векторный KNN с 3 метриками расстояния и весами `uniform`/`distance` |
| [`lab4_trees_ensembles.ipynb`](lab4_trees_ensembles.ipynb) | `CustomDecisionTree`, `CustomBagging`, `CustomStacking` | Дерево решений CART (Джини и MSE с квантильной дискретизацией порогов), бутстрап-агрегирование (Bagging) и Out-Of-Fold стекинг |
| [`lab5_clustering.ipynb`](lab5_clustering.ipynb) | `MyKMeans` | Алгоритм Ллойда (K-Means) с контролем сходимости центроидов |
| [`lab6_dimensionality_reduction.ipynb`](lab6_dimensionality_reduction.ipynb) | `CustomPCA` | Метод главных компонент через спектральное разложение ковариационной матрицы (`np.linalg.eigh`) |
| [`lab7_mlp_neural_networks.ipynb`](lab7_mlp_neural_networks.ipynb) | `Perceptron`, `MLPRegressorScratch` | Двухслойный перцептрон с прямым и обратным распространением ошибки (Backpropagation) |
