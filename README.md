# Lightweight Sybil Attack Detection Using Machine Learning with Game Theory Features

**Authors:** Manideep Sandireddy 
**Institution:** Vellore Institute of Technology, Chennai

## 📌 Project Overview
This project addresses the critical **accuracy-robustness trade-off** in Network Intrusion Detection Systems (NIDS). Standard ML models are brittle against adversarial attacks, while robust models often suffer from lower accuracy on clean traffic.

We propose a novel framework that models the attacker-defender interaction as a **Game-Theoretic Min-Max Game**. By solving for the **Nash Equilibrium**, we create a **Hybrid Ensemble Defender** that strategically balances high accuracy on normal traffic with high resilience against sophisticated attacks like **Projected Gradient Descent (PGD)**.

## 🚀 Key Features
* **Game-Theoretic Modeling:** Formulates the security problem as a non-zero-sum game between an Attacker and a Defender.
* **Adversarial Training:** Implements **FGSM** (Fast Gradient Sign Method) and **PGD** (Projected Gradient Descent) hardening loops in PyTorch.
* **Hybrid Ensemble:** Uses Nash Equilibrium probabilities to weigh model outputs, optimizing the "Maximin" strategy.
* **Real-Time Dashboard:** A Streamlit-based simulation (`dashboard.py`) demonstrating the system's viability in a live IIoT environment.

## 📊 Results
Our Game-Theoretic Hybrid Model successfully solves the trade-off:

| Model | Clean Accuracy | PGD Attack Accuracy | Status |
| :--- | :--- | :--- | :--- |
| **Standard Model** | **75.5%** | 32.7% (Collapse) | ❌ Fails |
| **PGD-Robust Model** | 73.9% | **70.8%** | ✅ Stable |
| **Hybrid** | **73.9%** | **69.0%** | ✅ Optimal |

## 🛠️ Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [ https://github.com/sandireddy4444/Light-weight-sybil-detection-using-ML-and-game-theory.git]
   cd Sybil-Detection-Game-Theory
2. Install dependencies:
    pip install -r requirements.txt
3. Run the Training Notebook: 
   Open notebooks/Main.ipynb in Jupyter or Google Colab to train the models and generate the game theory matrix.
4. Run the Dashboard Simulation:
   streamlit run src/dashboard.py

   
📂 Project Structure
data/: NSL-KDD Dataset files.

notebooks/: Core logic for training, game theory calculation, and evaluation.

src/: Application code for the live dashboard.

docs/: Full project report and research papers.


---

### 4. Git Commands to Upload

Once you have organized the folders and created the two files above, open your terminal (or Command Prompt) in that folder and run these commands one by one:

1.  **Initialize Git:**
    ```bash
    git init
    ```

2.  **Add all files:**
    ```bash
    git add .
    ```

3.  **Commit your work:**
    ```bash
    git commit -m "Initial commit: Full project upload with dashboard and game theory models"
    ```

4.  **Link to GitHub:**
    * Go to GitHub.com, create a new repository (e.g., `Sybil-Detection-Game-Theory`).
    * Copy the URL they give you (e.g., `https://github.com/yourname/repo.git`).
    * Run this command (replace the URL):
    ```bash
    git remote add origin https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git
    ```

5.  **Push to GitHub:**
    ```bash
    git branch -M main
    git push -u origin main
    ```

You are now live.
