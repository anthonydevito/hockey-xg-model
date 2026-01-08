# NHL Expected Goals (xG) Model: A Geometric Approach
**Developed for the Tampa Bay Lightning Data Science Initiative**

## 🏒 Project Overview
This project implements a **Shot-Based Expected Goals (xG) Model** using a gradient-boosted decision tree architecture (**XGBoost**). By utilizing spatial coordinates and game-state metadata, the model assigns a probability ($0$ to $1$) to every unblocked shot attempt, quantifying the quality of scoring chances.

This implementation emphasizes the **geometry of the ice** and the **physics of the game**, specifically focusing on shot angles, distances, and the impact of "rush" vs. "rebound" scenarios.

## 🛠️ Tech Stack & Engineering
As a former **Software Engineer**, I built this project with a "production-first" mindset:
* **Language:** Python 3.12
* **Modeling:** XGBoost, Scikit-learn
* **Data Processing:** Pandas, NumPy
* **DevOps:** Docker (Containerized for a reproducible research environment)

## 📐 Feature Engineering (The "Physics" Logic)
The core of this model is the transformation of raw coordinates into hockey-specific spatial features:
* **Distance to Net:** Calculated using the Euclidean distance from the arena-adjusted coordinates to the center of the goal $(89, 0)$.
* **Shot Angle:** The relative angle of the shooter to the net, identifying "low-danger" perimeter shots vs. "high-danger" slot opportunities.
* **Rebound Logic:** A binary feature identifying if a shot occurred within 3 seconds of a previous event, accounting for goalie displacement.
* **Rush Attempts:** Identifying shots generated through North-South puck movement where defensive gap control is compromised.

## 🚀 Getting Started

### Prerequisites
* [Docker](https://www.docker.com/) installed on your machine.

### Installation & Execution
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/anthonydevito/hockey-xg-model.git](https://github.com/anthonydevito/hockey-xg-model.git)
   cd hockey-xg-model

2. **Build the Docker Image:**
   ```bash
   docker build -t hockey-xg-model .

3. **Train the Model: The container is configured to run the training pipeline automatically and save the model to your local machine via a volume mount:**
   ```bash
   docker run -v $(pwd)/models:/app/models hockey-xg-model

## 📊 Performance & Validation
The model was trained on the **MoneyPuck Recent Seasons dataset** (~780k shots).
* **Key Driver:** Shot Distance remains the highest-weighted feature in the model's feature importance.
* **Insight:** "Rush" shots show a $1.4x$ increase in goal probability compared to stationary set plays from the same coordinates.

---
*Note: This project was developed as part of a technical demonstration for the Tampa Bay Lightning Data Science department.*