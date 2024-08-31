import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name(
    "ivan-solich-cloud-04c0b6435c30.json", scope)
client = gspread.authorize(creds)

sht = client.open_by_key("1fpS9jJEl2CnRY3rrI6W5xU1veAdTat8XsyE40QoYB6A")

worksheet3 = sht.worksheet("Hoja3")

masas = np.array([float(row[0]) for row in worksheet3.get('C3:C8')])
tensiones = 9.790969434 * masas  #Que serian las tensiones

longs = np.array([float(row[0]) for row in worksheet3.get('E3:E8')]) / 100


class Frecuency:
    def __init__(self, masas, tensiones, longs, densidad):
        self.masas = masas
        self.tensiones = tensiones
        self.longs = longs
        self.densidad = densidad
        self.g = 9.790969434
        self.tensiones_cuerda = longs * densidad * self.g
        self.tensiones_total = tensiones + self.tensiones_cuerda
        self.longs_cuadrado = None
        self.pendiente = None


    def modeloFrec(self, L=longs):
        return (1 / (2 * L)) * np.sqrt(self.tensiones_total / self.densidad)

    def modeloLong(self, T, f):
        return (1 / (2 * f)) * np.sqrt(T / self.densidad)

    def calculateSp(self):
        params, pcov = opt.curve_fit(self.modeloLong, self.tensiones, longs)

        print(params, pcov)

    def calculate(self):
        self.longs_cuadrado = self.longs ** 2
        def linear(x,a):
            return a*x

        self.pendiente, a_error = opt.curve_fit(linear, self.tensiones, self.longs_cuadrado)
        print(self.pendiente, a_error)

        return 1/(np.sqrt(4*self.pendiente*self.densidad))

    def propagation_error(self):

        sigma_L = 0.001
        sigma_m = 1e-5
        sigma_g = 0.01

        def propagation_error_T(sigma_m, sigma_g):
            sigma_T = np.sqrt((self.g * sigma_m) ** 2 + (self.masas * sigma_g) ** 2)
            return sigma_T

        def propagation_error_mu(sigma_L, sigma_m):
            # Propagación de errores para mu(L, m) = m / L
            sigma_mu = np.sqrt((sigma_m / self.longs) ** 2 + (self.masas * sigma_L / self.longs ** 2) ** 2)
            return sigma_mu

        sigma_T = propagation_error_T(sigma_m,sigma_g)
        sigma_mu = propagation_error_mu(sigma_L,sigma_m)

        # Derivadas parciales
        df_dL = (-1 / (2 * self.longs ** 2)) * np.sqrt(self.tensiones_total / self.densidad)
        df_dT = 1 / (4 * self.longs * np.sqrt(self.tensiones_total * self.densidad))
        df_dmu = -np.sqrt(self.tensiones_total) / ((4 * self.longs) * (self.densidad ** (3 / 2)))

        # Propagación de errores
        sigma_f = np.sqrt(
            (df_dL * sigma_L) ** 2 +
            (df_dT * sigma_T) ** 2 +
            (df_dmu * sigma_mu) ** 2
        )

        return sigma_f

    def plot(self):
        plt.scatter(self.tensiones_total, self.longs_cuadrado)
        plt.plot(self.tensiones_total, self.pendiente*self.tensiones_total)

        plt.show()



hilo_amarillo = Frecuency(masas, tensiones, longs, 0.0004508196721)
hilo_amarillo.calculateSp()
print(hilo_amarillo.modeloFrec())
print(hilo_amarillo.calculate())
print(hilo_amarillo.propagation_error())