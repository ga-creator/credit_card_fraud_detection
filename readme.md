# How to Use This Project

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fraud_detection.git
cd fraud_detection
```

---

## 2. Install the Required Packages

```bash
pip install -r requirements.txt
```

---

## 3. Download the Dataset

This project uses the **Credit Card Fraud Detection** dataset from Kaggle.

Download it here:

https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Extract the downloaded archive.

---

## 4. Add the Dataset

Create a folder named `data` in the project root (if it doesn't already exist) and place the downloaded `creditcard.csv` file inside it.

Your directory should look like this:

```
fraud_detection/
│
├── data/
│   └── creditcard.csv
├── src/
├── sample_data/
├── outputs/
├── app.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 5. Train the Models

Run:

```bash
python main.py
```

This will preprocess the data, train the models, and save the trained models and evaluation results.

---

## 6. Run the Application

Start the application with:

```bash
python app.py
```

Then open the local URL shown in the terminal to use the fraud detection interface.

---

## Notes

- The full dataset is **not included** in this repository because of its size.
- The `sample_data/` folder contains only a small sample for testing the expected input format and **cannot be used for training**.
