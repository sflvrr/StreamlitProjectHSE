import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import time
from concurrent.futures import ProcessPoolExecutor
import asyncio
import aiohttp
from datetime import datetime

MONTH_TO_SEASON = {12: "winter", 1: "winter", 2: "winter",
                   3: "spring", 4: "spring", 5: "spring",
                   6: "summer", 7: "summer", 8: "summer",
                   9: "autumn", 10: "autumn", 11: "autumn"}


def analyze_city_data(df_city):
    # Анализ данных для одного города.
    # Рассчитывает скользящее среднее и находит аномалии.
    # Сортировка по времени для корректного скользящего окна
    df_city = df_city.sort_values('timestamp')

    # Скользящее среднее и стандартное отклонение (окно 30 дней)
    df_city['rolling_mean'] = df_city['temperature'].rolling(window=30, min_periods=1).mean()
    df_city['rolling_std'] = df_city['temperature'].rolling(window=30, min_periods=1).std().fillna(0)

    # Тут будем аномалии рассматртривать, если выходим за пределы, например
    lower_bound = df_city['rolling_mean'] - 2 * df_city['rolling_std']
    upper_bound = df_city['rolling_mean'] + 2 * df_city['rolling_std']

    df_city['is_anomaly'] = (df_city['temperature'] < lower_bound) | (df_city['temperature'] > upper_bound)

    return df_city


def process_data_parallel(df):
    # Безжалостно распараллеливаем
    cities = df['city'].unique()
    city_dfs = [df[df['city'] == city].copy() for city in cities]

    # Используем ProcessPoolExecutor для параллельной обработки
    with ProcessPoolExecutor() as executor:
        results = list(executor.map(analyze_city_data, city_dfs))

    return pd.concat(results, ignore_index=True)


def get_weather_sync(city, api_key):
    # Синхронный запрос к OpenWeatherMap
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    return response.json()


async def fetch_weather_async(session, city, api_key):
    # А тут такой же асинхронный запросик
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    async with session.get(url) as response:
        return await response.json()


async def get_weather_async_wrapper(city, api_key):
    # Тут просто обертка для запуска асинхронного запроса в Streamlit
    async with aiohttp.ClientSession() as session:
        return await fetch_weather_async(session, city, api_key)


st.set_page_config(page_title="Анализ Температур", layout="wide")
st.title("🌤 Анализируем погодку и паттерны ее изменения")

if st.sidebar.button("Нужно сюда нажать!! ฅ^•ﻌ•^ฅ"):
    st.sidebar.image("https://cataas.com/cat", caption="Мяу! Держи котика для настроения.")
    st.toast("Котик успешно добавлен! 🐾")
st.sidebar.markdown("---")

st.sidebar.header("Настройки")

uploaded_file = st.sidebar.file_uploader("Загрузите файл с историческими данными (CSV)", type="csv")

if uploaded_file is not None:
    # Читаем файлики
    data = pd.read_csv(uploaded_file)
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    st.sidebar.success("Данные успешно загружены!")

    # Выбираем город
    cities = data['city'].unique()
    selected_city = st.sidebar.selectbox("Выберите город для анализа", cities)

    # Вводим API ключик (и покупаем ЧПУшечку)
    # Простите, у меня просто все рилсы в этих ЧПУшечках
    api_key = st.sidebar.text_input("Введите API-ключ OpenWeatherMap", type="password")

    # Анализируем данные и замеряем времечко-
    st.header(f"Анализ исторических данных: {selected_city}")

    start_time = time.time()
    # Запускаем параллельный анализ всего датасета
    processed_data = process_data_parallel(data)
    end_time = time.time()

    st.caption(f"Время выполнения параллельного анализа данных: {end_time - start_time:.4f} сек.")

    # Фильтруем данные только для выбранного города
    city_data = processed_data[processed_data['city'] == selected_city]

    # Тут сезонные профили делаем
    st.subheader("Сезонные профили")
    seasonal_stats = city_data.groupby('season')['temperature'].agg(['mean', 'std']).reset_index()
    seasonal_stats.columns = ['Сезон', 'Средняя температура (°C)', 'Стандартное отклонение (°C)']
    st.dataframe(seasonal_stats, use_container_width=True)

    # Тут график
    st.subheader("Временной ряд температур")

    fig = go.Figure()
    # Линия температуры
    fig.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['temperature'],
                             mode='lines', name='Температура', line=dict(color='lightblue')))
    # Скользящее среднее
    fig.add_trace(go.Scatter(x=city_data['timestamp'], y=city_data['rolling_mean'],
                             mode='lines', name='Скользящее среднее (30 дней)', line=dict(color='blue')))

    # Аномалии
    anomalies = city_data[city_data['is_anomaly']]
    fig.add_trace(go.Scatter(x=anomalies['timestamp'], y=anomalies['temperature'],
                             mode='markers', name='Аномалии', marker=dict(color='red', size=6)))

    fig.update_layout(xaxis_title="Дата", yaxis_title="Температура (°C)", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # Тут мониторим текущую темпу
    st.header("Текущая погода (OpenWeatherMap API)")

    if api_key:
        # Получение данных по API
        with st.spinner("Получение данных о погоде..."):
            weather_data = get_weather_sync(selected_city, api_key)

            # Ошибка 401!
            if str(weather_data.get("cod")) == "401":
                st.error(
                    '{"cod":401, "message": "Invalid API key. Please see https://openweathermap.org/faq#error401 for more info."}')
            elif str(weather_data.get("cod")) != "200":
                st.error(f"Ошибка API: {weather_data.get('message', 'Неизвестная ошибка')}")
            else:
                current_temp = weather_data['main']['temp']

                # Определяем сезон сейчас
                current_month = datetime.now().month
                current_season = MONTH_TO_SEASON[current_month]

                # Ищем исторические нормы для сезона сейчас
                season_hist = seasonal_stats[seasonal_stats['Сезон'] == current_season]
                if not season_hist.empty:
                    mean_temp = season_hist['Средняя температура (°C)'].values[0]
                    std_temp = season_hist['Стандартное отклонение (°C)'].values[0]

                    # Проверка на аномальность
                    lower_limit = mean_temp - 2 * std_temp
                    upper_limit = mean_temp + 2 * std_temp

                    is_current_anomaly = current_temp < lower_limit or current_temp > upper_limit

                    # Отображаем
                    cols = st.columns(3)
                    cols[0].metric("Текущая температура", f"{current_temp} °C")
                    cols[1].metric("Историческое среднее (сезон)", f"{mean_temp:.2f} °C")

                    if is_current_anomaly:
                        cols[2].error("Температура АНОМАЛЬНА для этого сезона!")
                    else:
                        cols[2].success("Температура в пределах сезонной нормы.")

                    st.info(
                        f"Диапазон нормы для сезона '{current_season}': от {lower_limit:.2f}°C до {upper_limit:.2f}°C.")
    else:
        st.warning("Пожалуйста, введите API-ключ в боковой панели для получения текущей погоды.")

else:
    st.info("Пожалуйста, загрузите файл `temperature_data.csv` в меню слева, чтобы начать.")