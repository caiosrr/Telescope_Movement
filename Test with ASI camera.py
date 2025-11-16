import requests
import json
import time

# --- CONFIGURAÇÃO ---
# Substitua pelo endereço IP que o seu ASCOM Remote Server mostrou.
# A câmera é o dispositivo 0 porque foi o primeiro que configuramos.
ALPACA_SERVER_ADDRESS = "http://127.0.0.1:11111/" 
CAMERA_NUMBER = 0

# Constrói a URL base para a nossa câmera específica
base_url = f"{ALPACA_SERVER_ADDRESS}api/v1/camera/{CAMERA_NUMBER}"

print(f"Tentando conectar ao servidor Alpaca em: {base_url}")

# 1. Obter o nome da câmera
try:
    # A "rota" para pegar o nome é simplesmente /name
    response = requests.get(f"{base_url}/setccdtemperature")
    if response.status_code == 200:
        data = response.json()
        if data.get("ErrorNumber", 0) == 0:
            temperature = data.get("Value")
            print(f"Temperature = {temperature}")
        else:
            print("Servidor retornou erro:", data["ErrorMessage"])
    else:
        print("Falha HTTP:", response.status_code)

except requests.exceptions.ConnectionError as e:
    print(f"ERRO DE CONEXÃO: Não foi possível alcançar o servidor.")
    print("Verifique se o endereço IP está correto e se o ASCOM Remote Server está rodando.")
    exit(1)
