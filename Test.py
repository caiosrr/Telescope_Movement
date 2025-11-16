import time
def calc_errors(d):
    """Calcula o erro considerando a circularidade no azimute."""
    error = [0,0]
    # erro com circularidade no azimute
    diff = (d) % 360
    if diff > 180:
        error[0] = diff - 360
    else:
        error[0] = diff
    return error

print(-355%360)  # Exemplo de uso