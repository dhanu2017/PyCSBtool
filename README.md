# PyCSBtool
**PyCSBtool** is a Python-based graphical user interface (GUI) designed to support the teaching and learning of basic control system concepts. It provides interactive tools for system modeling, response simulation, PID control, and system identification, making it ideal for undergraduate-level control systems courses.

---

## 🔧 Features

- **Frequency Domain Transfer Function Input**
- **State-Space Representation (A, B, C, D matrices)**
- **System Response Visualization** (Step, Impulse, Transient)
- **Pole-Zero Plot & Root Locus**
- **Interactive PID Controller Tuning** (Kp, Ki, Kd)
- **Parameter Sweep Analysis**
- **Serial Communication for COM Device Connection**
- **Basic System Identification (SysID) Tool**
- **Lightweight and Beginner-Friendly Interface**

---

## 📦 Installation

### Requirements

- Python 3.8 or above
- `matplotlib`
- `numpy`
- `scipy`
- `pyserial`
- `tkinter` (usually included with Python)

### Setup Instructions

```bash
# Step 1: Clone the repository
git clone https://github.com/yourusername/PyCSBtool.git
cd PyCSBtool

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the application
python pycsbtool.py

