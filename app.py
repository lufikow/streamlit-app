"""
app.py — Streamlit-приложение: анализ оттока клиентов фитнес-зала
Запуск:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import roc_curve, auc, accuracy_score

# Конфигурация страницы 
st.set_page_config(
    page_title="Gym Churn Analysis",
    page_icon="🏋️",
    layout="wide",
)

# Пути 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "gym_churn.csv")
MODEL_PATH = os.path.join(BASE_DIR, "data", "model_weights.mw")

# Загрузка данных и модели
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop("Phone", axis=1)
    return df

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

df = load_data()
bundle = load_model()
model = bundle["model"]
scaler = bundle["scaler"]
feature_names = bundle["feature_names"]

# Заголовок
st.title("🏋️ Анализ оттока клиентов фитнес-зала")
st.markdown(
    "Датасет содержит **4 000 записей** о клиентах. "
    "Цель — понять, кто уходит, и предсказать вероятность оттока."
)

# САЙДБАР — ФИЛЬТРЫ (контрол 1 и 2)
with st.sidebar:
    st.header("⚙️ Фильтры датасета")

    # Контрол 1 — Фильтр по полу
    gender_filter = st.selectbox(
        "Пол клиента",
        options=["Все", "Мужчины (1)", "Женщины (0)"],
        index=0,
        help="Фильтрует записи по признаку gender"
    )

    # Контрол 2 — Слайдер возраста
    age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
    age_range = st.slider(
        "Возраст клиентов",
        min_value=age_min,
        max_value=age_max,
        value=(age_min, age_max),
        help="Выберите диапазон возраста для анализа"
    )

# Применяем фильтры
df_view = df.copy()
if gender_filter == "Мужчины (1)":
    df_view = df_view[df_view["gender"] == 1]
elif gender_filter == "Женщины (0)":
    df_view = df_view[df_view["gender"] == 0]

df_view = df_view[(df_view["Age"] >= age_range[0]) & (df_view["Age"] <= age_range[1])]

st.caption(
    f"Показано записей: **{len(df_view)}** из {len(df)} "
    f"| Отток в выборке: **{df_view['Churn'].mean():.1%}**"
)

# ВКЛАДКИ

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Первичный анализ",
    "🔥 Корреляция",
    "📈 Распределения",
    "🤖 Прогноз оттока",
])


# ВКЛАДКА 1 — Первичный анализ

with tab1:
    st.subheader("KPI-метрики")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Всего клиентов", len(df_view))
    col2.metric("Ушли", int(df_view["Churn"].sum()))
    col3.metric("Остались", int((df_view["Churn"] == 0).sum()))
    col4.metric("Доля оттока", f"{df_view['Churn'].mean():.1%}")

    st.markdown("---")
    st.subheader("Общая статистика")
    st.dataframe(df_view.describe().T.style.format("{:.2f}"), width='stretch')

    st.markdown("---")
    st.subheader("Средние значения признаков: остались vs ушли")
    grp = df_view.groupby("Churn").mean().T
    grp.columns = ["Остались (0)", "Ушли (1)"]
    grp["Разница"] = grp["Ушли (1)"] - grp["Остались (0)"]
    st.dataframe(
        grp.style
           .background_gradient(subset=["Разница"], cmap="RdYlGn_r")
           .format("{:.3f}"),
        width='stretch',
    )
    st.write("Можно заметить некоторую разницу между средними значениями ушедших и оставшихся. \n"
             "Например, у оставшихся среднее время абонемента - 5.747 месяца, а у ушедших - 1.729 месяца. \n"
             "Большая разница видна в признаке дополнительных трат - Avg_additional_charges_total: \n"
             "Ушедшие явно имели меньший интерес к дополнительным покупкам.")


# ВКЛАДКА 2 — Корреляция

with tab2:
    st.subheader("Тепловая карта корреляций")
    fig, ax = plt.subplots(figsize=(12, 7))
    corr = df_view.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Матрица корреляций признаков", fontsize=13)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("Топ признаков по корреляции с оттоком (Churn)")
    churn_corr = corr["Churn"].drop("Churn").abs().sort_values(ascending=False)
    fig2, ax2 = plt.subplots(figsize=(8, 4))
    churn_corr.plot(kind="barh", ax=ax2, color=["#e74c3c" if v > 0.2 else "#95a5a6"
                                                  for v in churn_corr])
    ax2.set_xlabel("Абсолютная корреляция с Churn")
    ax2.axvline(0.2, color="red", linestyle="--", linewidth=1, label="порог 0.2")
    ax2.legend()
    plt.tight_layout()
    st.pyplot(fig2)


# ВКЛАДКА 3 — Распределения (Контрол 3 — выбор признака)

with tab3:
    st.subheader("Мы отрисуем гистограмму распределения признака по группам оттока (Churn)")

    # Контрол 3 — selectbox выбора признака
    numeric_cols = [c for c in df_view.columns if c != "Churn"]
    selected_feature = st.selectbox(
        "Выберите признак для анализа",
        options=numeric_cols,
        index=numeric_cols.index("Age"),
    )

    fig3, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for i, (churn_val, title) in enumerate([(0, "Остались (Churn=0)"), (1, "Ушли (Churn=1)")]):
        subset = df_view[df_view["Churn"] == churn_val][selected_feature]
        axes[i].hist(subset, bins=25, color=("#2ecc71" if churn_val == 0 else "#e74c3c"),
                     edgecolor="white", alpha=0.85)
        axes[i].set_title(title)
        axes[i].set_xlabel(selected_feature)
        axes[i].set_ylabel("Количество клиентов")
        axes[i].axvline(subset.mean(), color="black", linestyle="--",
                        linewidth=1.5, label=f"Среднее: {subset.mean():.2f}")
        axes[i].legend()
    plt.suptitle(f"Распределение «{selected_feature}»", fontsize=13, y=1.02)
    plt.tight_layout()
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("ROC-кривая модели")

    X_all = df[feature_names]
    y_all = df["Churn"]
    X_all_sc = scaler.transform(X_all)
    y_prob = model.predict_proba(X_all_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_all, y_prob)
    roc_auc_val  = auc(fpr, tpr)

    fig4, ax4 = plt.subplots(figsize=(6, 5))
    ax4.plot(fpr, tpr, "b", label=f"AUC = {roc_auc_val:.3f}", linewidth=2)
    ax4.plot([0, 1], [0, 1], "r--", linewidth=1)
    ax4.set_xlim([0, 1]); ax4.set_ylim([0, 1])
    ax4.set_xlabel("False Positive Rate"); ax4.set_ylabel("True Positive Rate")
    ax4.set_title("Receiver Operating Characteristic")
    ax4.legend(loc="lower right")
    plt.tight_layout()
    st.pyplot(fig4)


# ВКЛАДКА 4 — Прогноз оттока (ML)

with tab4:
    st.subheader("🤖 Предсказание вероятности оттока клиента")
    st.markdown(
        "Заполните характеристики клиента — наша модель рассчитает вероятность оттока."
    )

    col_a, col_b = st.columns(2)

    with col_a:
        gender = st.selectbox("Пол", [("Мужчина", 1), ("Женщина", 0)],
                                       format_func=lambda x: x[0])[1]
        near_location = st.selectbox("Живёте рядом с залом?",
                                       [("Да", 1), ("Нет", 0)],
                                       format_func=lambda x: x[0])[1]
        partner = st.selectbox("Являетесь партнёром/сотрудником компании?",
                                       [("Да", 1), ("Нет", 0)],
                                       format_func=lambda x: x[0])[1]
        promo_friends = st.selectbox("Пришли по промо от друга?",
                                       [("Да", 1), ("Нет", 0)],
                                       format_func=lambda x: x[0])[1]
        contract = st.selectbox("Период абонемента (мес.)",
                                       [1, 6, 12], index=0)
        group_visits = st.selectbox("Ходите на групповые занятия?",
                                       [("Да", 1), ("Нет", 0)],
                                       format_func=lambda x: x[0])[1]

    with col_b:
        age = st.slider("Возраст", 18, 60, 28)

        # Контрол 4 — числовой input для доп. расходов
        charges = st.number_input(
            "Суммарные доп. расходы (кофе, массаж, спа…)",
            min_value=0.0, max_value=600.0, value=100.0, step=10.0
        )
        months_left = st.slider("Месяцев до конца абонемента", 0, 12, 3)
        lifetime = st.slider("Срок членства (мес. с начала)", 0, 30, 3)
        freq_total = st.slider("Сред. тренировок/нед. (всего)", 0.0, 7.0, 2.0, 0.1)
        freq_month = st.slider("Сред. тренировок/нед. (прошлый мес.)", 0.0, 7.0, 2.0, 0.1)

    if st.button("🔮 Рассчитать вероятность оттока", width='stretch'):
        input_data = np.array([[
            gender, near_location, partner, promo_friends,
            contract, group_visits, age, charges,
            months_left, lifetime, freq_total, freq_month,
        ]])

        input_scaled = scaler.transform(input_data)
        prob_churn = model.predict_proba(input_scaled)[0][1]
        prediction = model.predict(input_scaled)[0]

        st.markdown("---")
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            if prediction == 1:
                st.error(f"⚠️ Высокий риск оттока\n\nВероятность: **{prob_churn:.1%}**")
            else:
                st.success(f"✅ Клиент останется\n\nВероятность оттока: **{prob_churn:.1%}**")

        with col_res2:
            # Прогресс-бар вероятности
            fig5, ax5 = plt.subplots(figsize=(5, 1.2))
            ax5.barh([""], [prob_churn], color="#e74c3c", height=0.5)
            ax5.barh([""], [1 - prob_churn], left=[prob_churn],
                     color="#2ecc71", height=0.5)
            ax5.set_xlim(0, 1)
            ax5.axvline(0.5, color="white", linewidth=2)
            ax5.set_xlabel("Вероятность оттока")
            ax5.set_title(f"Отток: {prob_churn:.1%}   |   Остаётся: {1-prob_churn:.1%}")
            ax5.xaxis.set_tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig5)

        # Топ-факторы риска для данного клиента
        st.markdown("**💡 Ключевые факторы риска по данным EDA:**")
        tips = []
        if freq_month < 1.5:
            tips.append("🔴 Мало тренировок в прошлом месяце (< 1.5/нед.) — главный триггер оттока")
        if months_left < 2:
            tips.append("🔴 До конца абонемента осталось менее 2 месяцев")
        if lifetime < 3:
            tips.append("🟡 Небольшой срок членства — клиент ещё не привязался к залу")
        if age < 28:
            tips.append("🟡 Молодой клиент: до 28 лет вероятность импульсивного ухода выше")
        if contract <= 1:
            tips.append("🟡 Короткий абонемент (1 мес.) — рекомендуется предложить скидку на год")
        if promo_friends == 0 and partner == 0:
            tips.append("🟢 Нет промо/скидок — попробуйте реферальную программу")
        if not tips:
            tips.append("✅ Профиль клиента не содержит явных факторов риска")
        for tip in tips:
            st.write(tip)

# ─── Футер ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Streamlit app by Dorokhin, 31STС.")
st.caption("Model: SVC (kernel=linear, C=1.5, GridSearchParams) | Scaler: StandardScaler | Dataset: gym_churn.csv")
