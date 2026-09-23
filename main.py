import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler


# Количество клиентов
n = 500

# Создаём случайные данные
age = np.random.randint(18, 70, n)
income = np.random.randint(30000, 250000, n)
credits = np.random.randint(0, 6, n)
late_payments = np.random.randint(0, 8, n)

# Собираем всё в таблицу
df = pd.DataFrame({
    'age': age,
    'income': income,
    'credits': credits,
    'late_payments': late_payments
})


# Считаем баллы риска
risk_score = (
    (df['income'] < 70000).astype(int) * 2 +
    ((df['income'] >= 70000) & (df['income'] < 120000)).astype(int) +
    (df['credits'] >= 4).astype(int) * 2 +
    ((df['credits'] >= 2) & (df['credits'] < 4)).astype(int) +
    (df['late_payments'] >= 4).astype(int) * 3 +
    ((df['late_payments'] >= 2) & (df['late_payments'] < 4)).astype(int)
)

# Переводим баллы в три уровня риска
df['risk'] = pd.cut(
    risk_score,
    bins=[-1, 2, 4, 100],
    labels=['low', 'medium', 'high']
)


# Нечёткая логика

# Определяем уровень дохода
def income_level(income):
    if income < 70000:
        return 'low'
    elif income < 120000:
        return 'medium'
    else:
        return 'high'


# Определяем уровень просрочек
def late_level(late_payments):
    if late_payments < 2:
        return 'low'
    elif late_payments < 4:
        return 'medium'
    else:
        return 'high'


# Определяем риск по правилам
def fuzzy_risk(income, late_payments):
    income = income_level(income)
    late = late_level(late_payments)

    # Правила
    if income == 'low' and late == 'high':
        return 'high'

    elif income == 'low' and late == 'medium':
        return 'high'

    elif income == 'low' and late == 'low':
        return 'medium'

    elif income == 'medium' and late == 'high':
        return 'high'

    elif income == 'medium' and late == 'medium':
        return 'medium'

    elif income == 'medium' and late == 'low':
        return 'low'

    elif income == 'high' and late == 'high':
        return 'medium'

    elif income == 'high' and late == 'medium':
        return 'low'

    else:
        return 'low'


# Получаем результат нечёткой логики для всех клиентов
df['fuzzy_risk'] = df.apply(
    lambda row: fuzzy_risk(row['income'], row['late_payments']),
    axis=1
)

print("\nПервые 10 клиентов:")
print(df[['income', 'late_payments', 'risk', 'fuzzy_risk']].head(10))


# Нейронная сеть

X = df[['age', 'income', 'credits', 'late_payments']]
y = df['risk']

# Делим данные на обучение и тест
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Приводим признаки к примерно одному масштабу
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Создаём модель
model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    max_iter=1000,
    random_state=42
)

# Обучаем модель
model.fit(X_train, y_train)

# Проверяем модель на тестовых данных
y_pred = model.predict(X_test)

print("\nПервые 10 прогнозов нейронной сети:")
print(y_pred[:10])


# Считаем метрики
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test, y_pred, average='weighted', zero_division=0
)
recall = recall_score(
    y_test, y_pred, average='weighted', zero_division=0
)
f1 = f1_score(
    y_test, y_pred, average='weighted', zero_division=0
)

print("\nОценка нейронной сети:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1-score:", f1)


# Проверяем нового клиента

print("\nВведите данные нового клиента:")

new_age = int(input("Возраст: "))
new_income = int(input("Доход: "))
new_credits = int(input("Количество кредитов: "))
new_late = int(input("Количество просрочек: "))


# Результат нечёткой логики
new_fuzzy = fuzzy_risk(new_income, new_late)


# Результат нейронной сети
new_client = pd.DataFrame([[
    new_age,
    new_income,
    new_credits,
    new_late
]], columns=['age', 'income', 'credits', 'late_payments'])

new_client_scaled = scaler.transform(new_client)

new_nn = model.predict(new_client_scaled)[0]

print("\nРезультат:")
print("Нечёткая логика:", new_fuzzy)
print("Нейронная сеть:", new_nn)


# Сохраняем таблицу
df.to_csv('clients.csv', index=False)

print("\nДанные сохранены в clients.csv")


# Строим графики

risk_numeric = df['risk'].map({
    'low': 1,
    'medium': 2,
    'high': 3
})


# Риск в зависимости от дохода
plt.figure()

plt.scatter(
    df['income'],
    risk_numeric,
    alpha=0.4,
    s=20
)

plt.xlabel('Доход')
plt.ylabel('Кредитный риск')
plt.title('Зависимость кредитного риска от дохода')

plt.yticks(
    [1, 2, 3],
    ['Низкий', 'Средний', 'Высокий']
)

plt.savefig('risk_income.png')


# Риск в зависимости от просрочек
plt.figure()

plt.scatter(
    df['late_payments'],
    risk_numeric,
    alpha=0.4,
    s=20
)

plt.xlabel('Количество просрочек')
plt.ylabel('Кредитный риск')
plt.title('Зависимость кредитного риска от количества просрочек')

plt.yticks(
    [1, 2, 3],
    ['Низкий', 'Средний', 'Высокий']
)

plt.savefig('risk_late_payments.png')