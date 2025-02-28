## **Setting Up a Virtual Environment**
A virtual environment keeps your dependencies isolated from the system-wide Python installation.

### **1. Create a Virtual Environment**
Run the following command in the project folder:
```sh
python -m venv venv
```
This will create a folder named `venv` that contains the isolated environment.

---

### **2️. Activate the Virtual Environment**
#### **Windows (Command Prompt)**
```sh
venv\Scripts\activate
```
#### **Windows (PowerShell)**
```sh
venv\Scripts\Activate.ps1
```
#### **Mac/Linux (Terminal)**
```sh
source venv/bin/activate
```
After activation, your terminal should show `(venv)` at the beginning of the line.

---

### **3. Install Dependencies**
Once the virtual environment is activated, install all required dependencies using:
```sh
pip install -r requirements.txt
```
This will install **TensorFlow, Matplotlib, OpenCV, and any other required libraries**.
