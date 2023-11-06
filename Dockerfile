# Python Version -- 3.11v
FROM python:3.11

# Conatiner's Root Directory
WORKDIR /app/myNetworkPrototypes

# Copy Modules Auto Installation Text File into Container
COPY requirements.txt .

# PIP Install Dependencies from requirements.txt
RUN pip install -r requirements.txt

# Containers Profile Report Directory
WORKDIR /app/myNetworkPrototypes/reports

# Copy Report HTML File(PLEASE NOTE: _blank until Network Script is Successfully Run and Profile Report has generated the HTML Code!)
COPY myNetworkPrototypes/reports/diabetes_report.html /app/myNetworkPrototypes/reports/

# Conatiner's Datasets Directory
WORKDIR /app/myNetworkPrototypes/datasets

# Copy the Diabetes Dataset
COPY myNetworkPrototypes/datasets/diabetes.csv /app/myNetworkPrototypes/datasets/

# Containers Network Directory
WORKDIR /app/myNetworkPrototypes/network_prototype.v3/Network.v3(stable)

# Copy Network Script into Working Directory
COPY myNetworkPrototypes/network_prototype.v3/Network.v3(stable)/latest__network.prototype.v3.0.py /app/myNetworkPrototypes/network_prototype.v3/Network.v3(stable)/

# CLI Command - Run Python Script('semantic.py')
CMD ["python", "latest__network.prototype.v3.0.py"]