# 🌱 Predicting Irrigation Need | Classification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![License](https://img.shields.io/github/license/tugcesi/predicting-irrigation-need)

Toprak, hava durumu ve bitki verilerini kullanarak tarla sulama ihtiyacını tahmin eden bir makine öğrenmesi projesi.

---

## 📌 Proje Özeti

Kaggle Playground Series veri seti (630.000 kayıt) üzerinde çoklu ML sınıflandırıcıları eğitilmiş ve **en iyi model** Streamlit arayüzüne entegre edilmiştir.

**Hedef Değişken:** `Irrigation_Need` → `Low` / `Medium` / `High`

---

## 📊 Özellikler (19 Feature)

| Grup | Özellikler |
|------|-----------|
| 🌍 Toprak | Soil_Type, Soil_pH, Soil_Moisture, Organic_Carbon, Electrical_Conductivity |
| 🌤️ Hava | Temperature_C, Humidity, Rainfall_mm, Sunlight_Hours, Wind_Speed_kmh |
| 🌾 Bitki | Crop_Type, Crop_Growth_Stage, Season |
| 🚜 Tarla | Irrigation_Type, Water_Source, Field_Area_hectare, Mulching_Used, Previous_Irrigation_mm, Region |

---

## 🤖 Kullanılan Algoritmalar

- Logistic Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest ✅
- AdaBoost / Gradient Boosting
- Gaussian / Bernoulli Naive Bayes
- XGBoost / LightGBM / CatBoost

---

## 🚀 Kurulum & Çalıştırma

### 1. Repoyu Klonla
```bash
git clone https://github.com/tugcesi/predicting-irrigation-need.git
cd predicting-irrigation-need
```

### 2. Gereksinimleri Yükle
```bash
pip install -r requirements.txt
```

### 3. Modeli Eğit
```bash
python train_model.py
```
> `irrigation_model.joblib`, `scaler.joblib`, `feature_names.joblib` oluşturulur.

### 4. Uygulamayı Başlat
```bash
streamlit run app.py
```

---

## 📁 Dosya Yapısı

```
predicting-irrigation-need/
├── app.py                              # Streamlit uygulaması
├── train_model.py                      # Model eğitim scripti
├── irrigation_model.joblib             # Eğitilmiş model
├── scaler.joblib                       # StandardScaler
├── feature_names.joblib                # Feature sütun listesi
├── predicting-irrigation-need.ipynb    # EDA & model karşılaştırması
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🛠️ Kullanılan Teknolojiler

| Teknoloji | Kullanım |
|-----------|----------|
| Python | Temel dil |
| Scikit-learn | Model eğitimi |
| XGBoost / LightGBM / CatBoost | Boosting modelleri |
| Streamlit | Web arayüzü |
| Pandas / NumPy | Veri işleme |
| Joblib | Model kaydetme/yükleme |

---

## 📄 Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.
