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


# Step 1: Clone the repository
git clone https://github.com/yourusername/PyCSBtool.git
cd PyCSBtool

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the application
python CT_V20.py


### How to Use PyCSBtool
🚀 How to Use PyCSBtool
1. System Modeling
Choose Frequency Domain Transfer Function to enter a transfer function (e.g., G = 1/(s^2+10s+20)).

Or, select Time Domain State-Space Representation to input matrices A, B, C, and D manually.

2. System Visualization Tools
Use the Pole Zero and Root Locus buttons to display system characteristics in the frequency domain.

Click Step, Impulse, or Transient to simulate corresponding system responses.

Adjust Time Axis and graph theme for better readability.

3. PID Controller Tuning
Adjust the Kp, Ki, and Kd sliders or enter values manually.

Observe how system response changes with different controller parameters.

Use this for real-time understanding of control action and stability.

4. Parameter Sweep
Enter a range of gain values to analyze how the system reacts across multiple controller settings.

Click Sweep to visualize sensitivity and performance trends.

5. Connect COM Device (Optional)
Select the correct COM Port and Baud Rate to connect to a microcontroller (e.g., Arduino).

Click Connect to establish a serial link.

Useful for integrating hardware in lab experiments.

6. System Identification
Use the SysID section to input experimental input-output data and estimate system models.

View estimated transfer functions or state-space equivalents.

7. Save/Load
Use Save to store current configurations (model, PID gains, etc.).

Use Load to retrieve a saved session and continue analysis.

8. Exit
Click Exit to close the application safely.

