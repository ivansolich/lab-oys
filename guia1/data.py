import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import gspread
from oauth2client.service_account import ServiceAccountCredentials
from openpyxl import Workbook

scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name(
    "ivan-solich-cloud-f74d57e90205.json", scope)
client = gspread.authorize(creds)

sht = client.open_by_key("1fpS9jJEl2CnRY3rrI6W5xU1veAdTat8XsyE40QoYB6A")

worksheet3 = sht.worksheet("Hoja3")


class Frecuency:
    def __init__(self, masas, longs_h, longs_v, densidad):
        self.g = 9.790969434
        self.masas = masas
        self.tensiones = self.masas * self.g
        self.longs_h = longs_h
        self.longs_v = longs_v
        self.densidad = densidad
        self.tensiones_total = self.tensiones + self.longs_v * self.densidad * self.g
        self.longs_cuadrado = None
        self.pendiente = None
        self.a_error = None
        self.sigma_L = 0.001
        self.sigma_m = 1e-5
        self.sigma_g = 0.01

    def modeloFrec(self):
        return (1 / (2 * self.longs_h)) * np.sqrt(self.tensiones_total / self.densidad)

    def modeloLong(self, T, f):
        return (1 / (2 * f)) * np.sqrt(T / self.densidad)

    def calculateSp(self):
        params, pcov = opt.curve_fit(self.modeloLong, self.tensiones_total, self.longs_h)

        return params, pcov

    def propagation_error_mu(self, sigma_L, sigma_m):
        # Propagación de errores para mu(L, m) = m / L
        sigma_mu = np.sqrt((sigma_m / 0.976) ** 2 + (0.00044 * sigma_L / 0.976 ** 2) ** 2)
        return sigma_mu

    def frec_slope(self):
        self.longs_cuadrado = self.longs_h ** 2

        def linear(x, a):
            return a * x

        self.pendiente, self.a_error = opt.curve_fit(linear, self.tensiones, self.longs_cuadrado)

        return 1 / (np.sqrt(4 * self.pendiente * self.densidad))

    def error_frec_slope(self):
        sigma_mu = self.propagation_error_mu(self.sigma_L, self.sigma_m)
        sigma_a = self.a_error[0]

        # Derivadas parciales
        df_da = -self.densidad / (2 * self.pendiente * np.sqrt(4 * self.pendiente * self.densidad))
        df_dmu = -self.pendiente / (2 * self.densidad * np.sqrt(4 * self.pendiente * self.densidad))

        # Propagación de errores
        sigma_f = np.sqrt(
            (df_da * sigma_a) ** 2 +
            (df_dmu * sigma_mu) ** 2
        )

        return sigma_f

    def propagation_error_T(self):
        sigma_mu = self.propagation_error_mu(self.sigma_L, self.sigma_m)

        dT_dg = self.masas + self.longs_v * self.densidad
        dT_dm = self.g
        dT_dL = self.densidad * self.g
        dT_dmu = self.longs_v * self.g

        # Propagación de errores
        sigma_T = np.sqrt(
            (dT_dg * self.sigma_g) ** 2 +
            (dT_dm * self.sigma_m) ** 2 +
            (dT_dL * self.sigma_L) ** 2 +
            (dT_dmu * sigma_mu) ** 2
        )

        return sigma_T

    def propagation_error(self):
        sigma_mu = self.propagation_error_mu(self.sigma_L, self.sigma_m)

        sigma_T = self.propagation_error_T()
        """print(self.tensiones_total, "tensiones total")
        print(sigma_T, "errores T")"""

        # Derivadas parciales
        df_dL = (-1 / (2 * self.longs_h ** 2)) * np.sqrt(self.tensiones_total / self.densidad)
        df_dT = 1 / (4 * self.longs_h * np.sqrt(self.tensiones_total * self.densidad))
        df_dmu = -np.sqrt(self.tensiones_total) / ((4 * self.longs_h) * (self.densidad ** (3 / 2)))

        # Propagación de errores
        sigma_f = np.sqrt(
            (df_dL * self.sigma_L) ** 2 +
            (df_dT * sigma_T) ** 2 +
            (df_dmu * sigma_mu) ** 2
        )

        return sigma_f

    def weighted_average(self):
        weights = 1 / (self.propagation_error() ** 2)

        mean_weighted = np.average(self.modeloFrec(), weights=weights)

        # Calcular la varianza ponderada
        weighted_variance = np.sum(weights * (self.modeloFrec() - mean_weighted) ** 2) / np.sum(weights)

        # Número de mediciones
        N = len(self.modeloFrec())

        # Calcular el error estándar ponderado
        error_estandar_weighted = np.sqrt(weighted_variance / N)

        return mean_weighted, error_estandar_weighted

    def plot(self):
        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'
        #plt.rcParams.update({'font.size': 13})

        plt.style.use('seaborn-v0_8-deep')

        plt.scatter(self.tensiones_total, self.longs_cuadrado)
        plt.plot(self.tensiones_total, self.pendiente * self.tensiones_total)
        #plt.errorbar(self.tensiones_total, self.longs_cuadrado, self.sigma_L ** 2,self.propagation_error_T(),fmt="None",capsize=3)

        plt.xlabel("Tensiones [N]")
        plt.ylabel(f"Longitudes$^2$ [$m^2$]")

        plt.tight_layout()

        plt.savefig("img1.png", dpi=300)

        plt.show()

    def plot_comp_series(self):
        frec, error = self.weighted_average()

        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

        plt.style.use('seaborn-v0_8-deep')

        plt.errorbar(range(1, 7), self.modeloFrec(), yerr=self.propagation_error(), fmt='o', capsize=3,
                     label='Serie de frecuencias')
        plt.hlines(y=frec, xmin=0, xmax=7, colors="black", label='Frecuencia Media', alpha=0.5)
        plt.fill_between(range(0, 8), frec + error, frec - error, alpha=0.3, color='tab:orange', label='Area de error')

        plt.xlim(0.5, 6.5)
        plt.ylabel("Frecuencia [Hz]")
        plt.legend()

        plt.tight_layout()

        plt.savefig("img2.png", dpi=300)

        plt.show()

    def final_plot(self):
        frec_avearege, error_average = self.weighted_average()
        frecSp, error_sp = self.calculateSp()

        frecs = [frec_avearege, frecSp, self.frec_slope()]
        errors = [error_average, error_sp, self.error_frec_slope()]

        plt.rcParams['mathtext.fontset'] = 'stix'
        plt.rcParams['font.family'] = 'STIXGeneral'

        plt.style.use('seaborn-v0_8-deep')

        plt.errorbar([1], frecs[0], yerr=errors[0], fmt='o', capsize=3, label='Media Ponderada')
        plt.errorbar([2], frecs[1], yerr=errors[1], fmt='o', capsize=3, label='curve_fit')
        plt.errorbar([3], frecs[2], yerr=errors[2], fmt='o', capsize=3, label='Pendiente')

        labels = ["", "1", "2", "3"]
        plt.xticks(ticks=range(4), labels=labels, ha="right")

        plt.xlim(0.5, 3.5)
        plt.ylabel("Frecuencia [Hz]")
        plt.legend(title='Métodos')

        plt.tight_layout()

        plt.savefig("img3.png", dpi=300)

        plt.show()


masas_amarillo = np.array([float(row[0]) for row in worksheet3.get('C3:C8')])
longs_h_amarillo = np.array([float(row[0]) for row in worksheet3.get('E3:E8')]) / 100
longs_v_amarillo = np.array([float(row[0]) for row in worksheet3.get('H3:H8')]) / 100

hilo_amarillo = Frecuency(masas_amarillo, longs_h_amarillo, longs_v_amarillo, 0.0004508196721)

masas_azul = np.array([float(row[0]) for row in worksheet3.get('C12:C15')])
longs_h_azul = np.array([float(row[0]) for row in worksheet3.get('E12:E15')]) / 100
longs_v_azul = np.array([float(row[0]) for row in worksheet3.get('H12:H15')]) / 100

lana_azul = Frecuency(masas_azul, longs_h_azul, longs_v_azul, 0.0004293193717)
a, b = lana_azul.calculateSp()
print(a, b[0])
print(lana_azul.frec_slope())
print(lana_azul.error_frec_slope())
print(lana_azul.modeloFrec())
print(lana_azul.propagation_error())
print(lana_azul.weighted_average())
