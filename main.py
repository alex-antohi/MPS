# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
from datetime import datetime
import pandas as pd


class Patient:
    def __init__(self, sex, varsta, data_simpt, simpt_decl,
                 data_internare, simpt_rap, diagnostic, calatorii, mij_trans,
                 contact, data_rez, rez):
        self.sex = sex
        self.varsta = varsta
        self. data_simpt = data_simpt
        self.simpt_decl = simpt_decl
        self.data_internare = data_internare
        self.simpt_rap = simpt_rap
        self.diagnostic = diagnostic
        self.calatorii = calatorii
        self.mij_trans = mij_trans
        self.contact = contact
        self.data_rez = data_rez
        self.rez = rez

    if __name__ == '__main__':
        # df = pd.read_excel(r'C:\Users\alexa\PycharmProjects\pythonProject\mps.dataset.xlsx')
        df = pd.read_excel('mps.dataset.xlsx')
        df = pd.DataFrame(df, columns=df.columns)
        for i in df.index:

            # codificare instituția sursă
            if isinstance(df['instituția sursă'][i], str):
                if 'X' in df['instituția sursă'][i].upper():
                    df['instituția sursă'][i] = 0
                elif 'Y' in df['instituția sursă'][i].upper():
                    df['instituția sursă'][i] = 1
                elif 'Z' in df['instituția sursă'][i].upper():
                    df['instituția sursă'][i] = 2
            else:
                df['sex'][i] = -1

            # codificare sex
            if isinstance(df['sex'][i], str):
                if 'MASCULIN' in df['sex'][i].upper() or 'M' in df['sex'][i].upper():
                    df['sex'][i] = 0
                elif 'FEMININ' in df['sex'][i].upper() or 'F' in df['sex'][i].upper():
                    df['sex'][i] = 1
                else:
                    print(df['sex'][i], i)
            else:
                df['sex'][i] = -1

            # codificare varsta
            if isinstance(df['vârstă'][i], int):
                df['vârstă'][i] = float(df['vârstă'][i])
            elif isinstance(df['vârstă'][i], str):
                if df['vârstă'][i].isdecimal():
                    df['vârstă'][i] = float(df['vârstă'][i])
                else:
                    if 'ANI' in df['vârstă'][i].upper():
                        df['vârstă'][i] = df['vârstă'][i].upper().replace(' ANI', '')
                        if df['vârstă'][i].isdecimal():
                            df['vârstă'][i] = float(df['vârstă'][i])
                        elif 'LUN' in df['vârstă'][i].upper():
                            df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNA', '')
                            df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNI', '')
                            df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNA', '')
                            df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNI', '')
                            df['vârstă'][i] = float(df['vârstă'][i].split(' ')[0]) +\
                                float(df['vârstă'][i].split(' ')[1]) / 12
                    elif 'LUN' in df['vârstă'][i].upper():
                        df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNA', '')
                        df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNI', '')
                        df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNA', '')
                        df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNI', '')
                        if df['vârstă'][i].isdecimal():
                            df['vârstă'][i] = float(df['vârstă'][i])
                        else:
                            df['vârstă'][i] = df['vârstă'][i].upper().replace(' ', '')
                            if df['vârstă'][i].isdecimal():
                                df['vârstă'][i] = float(df['vârstă'][i])
                    # valori neabordate : nou nascut / 1 zi etc.
                    # else:
                    #    print(df['vârstă'][i], i)

            # codificare dată debut simptome declarate ca numarul zilei din an
            if isinstance(df['dată debut simptome declarate'][i], datetime):
                df['dată debut simptome declarate'][i] = float(df['dată debut simptome declarate'][i].strftime("%j"))
            elif isinstance(df['dată debut simptome declarate'][i], str):
                if 'NU' in df['dată debut simptome declarate'][i].upper() or\
                        '-' in df['dată debut simptome declarate'][i]:
                    df['dată debut simptome declarate'][i] = -1
                else:
                    print(df['dată debut simptome declarate'][i])
            else:
                if math.isnan(df['dată debut simptome declarate'][i]):
                    df['dată debut simptome declarate'][i] = -1
