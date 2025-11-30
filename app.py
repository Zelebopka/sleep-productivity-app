import gradio as gr
import numpy as np
import joblib
import json

try:
    model = joblib.load('best_sleep_classifier.pkl')
    scaler = joblib.load('best_scaler.pkl')
    label_encoder = joblib.load('best_label_encoder.pkl')
    
    with open('best_model_info.json', 'r') as f:
        model_info = json.load(f)
    
    print(f" Тип модели: {model_info.get('model_type', 'Unknown')}")
    
except Exception as e:
    print(f" Ошибка загрузки: {e}")
    raise

def predict_productivity(age, gender, total_sleep, sleep_quality, exercise,
                        caffeine, screen_time, work_hours, mood, stress_level):

    try:
        if (age == 0 and total_sleep == 0 and sleep_quality == 0 and
            exercise == 0 and caffeine == 0 and screen_time == 0 and
            work_hours == 0 and mood == 0 and stress_level == 0):

            return """
# 🎸 ВНЕЗАПНЫЙ РИКРОЛЛ! 🎸

## Never Gonna Give You Up! 🎵

![Rickroll GIF](https://media.giphy.com/media/Vuw9m5wXviFIQ/giphy.gif)

**Ты либо угараешь либо тебя не существует ...
А теперь введите реальные данные для предсказания продуктивности...**
"""

        if (age == 40 and work_hours == 12 and
            total_sleep == 0 and sleep_quality == 0 and exercise == 0 and
            caffeine == 0 and screen_time == 0 and mood == 0 and stress_level == 0):

            return """
# 🔥 RIP AND TEAR! 🔥

## The Only Thing They Fear...

![Doom GIF](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExajkzZDk2eXQxeDF0azRodGNiZTdiZ2k0MzAxNTZob2tqNDR1dHo4MCZlcD12MV9naWZzX3NlYXJjaCZjdD1n/XZbAsygv5VTZ5oumfo/giphy.gif)

**Is You **

*P.S. Может все-таки....
поспишь?*
"""

        good_sleep_conditions = sleep_quality >= 7 and total_sleep >= 6.5 and stress_level <= 4
        bad_sleep_conditions = sleep_quality <= 3 or total_sleep <= 4.5 or stress_level >= 7

        if good_sleep_conditions:
            base_tendency = "HIGH"
        elif bad_sleep_conditions:
            base_tendency = "LOW"
        else:
            base_tendency = "NEUTRAL"

        gender_encoded = label_encoder.transform([gender])[0]

        base_features = [
            age, gender_encoded, total_sleep, sleep_quality,
            exercise, caffeine, screen_time, work_hours,
            mood, stress_level
        ]

        sleep_start_hour = 22
        sleep_end_hour = 6

        if sleep_end_hour < sleep_start_hour:
            actual_sleep_duration = (24 - sleep_start_hour) + sleep_end_hour
        else:
            actual_sleep_duration = sleep_end_hour - sleep_start_hour

        sleep_efficiency = sleep_quality / (total_sleep + 0.1)
        work_sleep_ratio = work_hours / (total_sleep + 0.1)
        caffeine_per_hour = caffeine / 16
        stress_mood_interaction = stress_level * mood
        exercise_productivity = exercise * sleep_quality
        late_sleeper = 1 if sleep_start_hour > 23 else 0
        early_riser = 1 if sleep_end_hour < 6 else 0

        if age <= 25: age_group = 0
        elif age <= 35: age_group = 1
        elif age <= 45: age_group = 2
        elif age <= 55: age_group = 3
        else: age_group = 4

        all_features = base_features + [
            actual_sleep_duration, sleep_efficiency, work_sleep_ratio,
            caffeine_per_hour, stress_mood_interaction, exercise_productivity,
            late_sleeper, early_riser, age_group,
            sleep_start_hour, sleep_end_hour
        ]

        input_scaled = scaler.transform([all_features])
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]

        model_confidence = probability[1] if prediction == 1 else probability[0]

        final_prediction = prediction
        final_confidence = model_confidence
        logic_used = " (модель)"

        if base_tendency == "HIGH" and prediction == 0 and model_confidence < 0.6:
            final_prediction = 1
            final_confidence = 0.7
            logic_used = " (логика: хорошие условия сна)"
        elif base_tendency == "LOW" and prediction == 1 and model_confidence < 0.6:
            final_prediction = 0
            final_confidence = 0.7
            logic_used = " (логика: плохие условия сна)"

        result_class = "Высокая продуктивность 🚀" if final_prediction == 1 else "Низкая продуктивность 😴"

        output_text = f"""🎯 РЕЗУЛЬТАТ ПРЕДСКАЗАНИЯ:


{result_class}

📊 Уверенность: {final_confidence:.2%}

🔍 Источник: {logic_used}


📈 АНАЛИЗ ВХОДНЫХ ДАННЫХ:

• Возраст: {age} лет

• Пол: {gender}

• Сон: {total_sleep} часов (качество: {sleep_quality}/10)

• Стресс: {stress_level}/10

• Настроение: {mood}/10

• Работа: {work_hours} часов/день

• Активность: {exercise} мин/день"""

        return output_text

    except Exception as e:
        return f"❌ Ошибка предсказания: {str(e)}"

iface = gr.Interface(
    fn=predict_productivity,
    inputs=[
        gr.Number(label="Age", value=30, minimum=0, maximum=80),
        gr.Dropdown(["Male", "Female"], label="Gender", value="Male"),
        gr.Slider(0, 12, label="Total Sleep Hours", value=7.5),
        gr.Slider(0, 10, label="Sleep Quality", value=8),
        gr.Number(label="Exercise (mins/day)", value=45, minimum=0, maximum=180),
        gr.Number(label="Caffeine Intake (mg)", value=100, minimum=0, maximum=500),
        gr.Number(label="Screen Time Before Bed (mins)", value=60, minimum=0, maximum=240),
        gr.Slider(0, 12, label="Work Hours (hrs/day)", value=8.0),
        gr.Slider(0, 10, label="Mood Score", value=7),
        gr.Slider(0, 10, label="Stress Level", value=4)
    ],
    outputs=gr.Markdown(),
    title="😴 Sleep Quality and Productivity Predictor 🚀",
    description="Введите параметры сна и образа жизни для предсказания продуктивности на следующий день",
    allow_flagging="never"
)

print("Запускаем приложение...")
print("Пасхалки:")
print("   • Все нули")
print("   • Возраст 40, работа 12 часов, остальные нули")
iface.launch(share=True)