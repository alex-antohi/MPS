# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
from datetime import datetime
import pandas as pd
import re
from texttable import Texttable


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

        t = Texttable()

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
                df['instituția sursă'][i] = -1

            # codificare sex
            if isinstance(df['sex'][i], str):
                if 'MASCULIN' in df['sex'][i].upper() or 'M' in df['sex'][i].upper():
                    df['sex'][i] = 0
                elif 'FEMININ' in df['sex'][i].upper() or 'F' in df['sex'][i].upper():
                    df['sex'][i] = 1
                else:
                    # exceptii
                    print(df['sex'][i], i)
            else:
                df['sex'][i] = -1

            # codificare varsta
            if isinstance(df['vârstă'][i], int):
                df['vârstă'][i] = int(df['vârstă'][i])
            elif isinstance(df['vârstă'][i], str):
                if df['vârstă'][i].isdecimal():
                    df['vârstă'][i] = int(df['vârstă'][i])
                else:
                    if 'ANI' in df['vârstă'][i].upper():
                        df['vârstă'][i] = df['vârstă'][i].upper().replace(' ANI', '')
                        if df['vârstă'][i].isdecimal():
                            df['vârstă'][i] = int(df['vârstă'][i])
                        elif 'LUN' in df['vârstă'][i].upper():
                            df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNA', '')
                            df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNI', '')
                            df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNA', '')
                            df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNI', '')
                            df['vârstă'][i] = int(df['vârstă'][i].split(' ')[0])
                            #   + int(df['vârstă'][i].split(' ')[1]) / 12
                    elif 'LUN' in df['vârstă'][i].upper():
                        df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNA', '')
                        df['vârstă'][i] = df['vârstă'][i].upper().replace(' LUNI', '')
                        df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNA', '')
                        df['vârstă'][i] = df['vârstă'][i].upper().replace('LUNI', '')
                        if df['vârstă'][i].isdecimal():
                            df['vârstă'][i] = int(df['vârstă'][i])
                        else:
                            df['vârstă'][i] = df['vârstă'][i].upper().replace(' ', '')
                            if df['vârstă'][i].isdecimal():
                                df['vârstă'][i] = int(df['vârstă'][i])
                    else:
                        # exceptii
                        # print(df['vârstă'][i], i)
                        df['vârstă'][i] = 0
            else:
                if math.isnan(df['vârstă'][i]):
                    df['vârstă'][i] = -1

            # codificare dată debut simptome declarate ca numarul zilei din an
            if isinstance(df['dată debut simptome declarate'][i], datetime):
                # print(df['dată debut simptome declarate'][i])
                df['dată debut simptome declarate'][i] = int(df['dată debut simptome declarate'][i].strftime("%j"))
            elif isinstance(df['dată debut simptome declarate'][i], str):
                if 'NU' in df['dată debut simptome declarate'][i].upper() or\
                        '-' in df['dată debut simptome declarate'][i]:
                    df['dată debut simptome declarate'][i] = -1
                else:
                    df['dată debut simptome declarate'][i] = df['dată debut simptome declarate'][i].replace(' ', '-')
                    df['dată debut simptome declarate'][i] = df['dată debut simptome declarate'][i].replace('.', '-')
                    df['dată debut simptome declarate'][i] = df['dată debut simptome declarate'][i].replace(',', '-')

                    if re.match(r"(\d+)-(\d+)-(\d\d\d\d)$", df['dată debut simptome declarate'][i]):
                        df['dată debut simptome declarate'][i] = \
                            datetime.strptime(df['dată debut simptome declarate'][i], '%d-%m-%Y')
                        df['dată debut simptome declarate'][i] = \
                            int(df['dată debut simptome declarate'][i].strftime("%j"))
                    else:
                        # exceptii nerezolvate
                        df['dată debut simptome declarate'][i] = 0
                        # print(df['dată debut simptome declarate'][i])
            else:
                if math.isnan(df['dată debut simptome declarate'][i]):
                    df['dată debut simptome declarate'][i] = -1

            # codificare dată internare ca numarul zilei din an
            if isinstance(df['dată internare'][i], datetime):
                # print(df['dată internare'][i])
                df['dată internare'][i] = int(df['dată internare'][i].strftime("%j"))
            elif isinstance(df['dată internare'][i], str):
                if 'NU' in df['dată internare'][i].upper() or\
                        '-' in df['dată internare'][i]:
                    df['dată internare'][i] = -1
                else:
                    df['dată internare'][i] = df['dată internare'][i].replace(' ', '-')
                    df['dată internare'][i] = df['dată internare'][i].replace('.', '-')
                    df['dată internare'][i] = df['dată internare'][i].replace(',', '-')

                    if re.match(r"(\d+)-(\d+)-(\d\d\d\d)$", df['dată internare'][i]):
                        df['dată internare'][i] = datetime.strptime(df['dată internare'][i], '%d-%m-%Y')
                        df['dată internare'][i] = int(df['dată internare'][i].strftime("%j"))
                    else:
                        # exceptii nerezolvate
                        df['dată internare'][i] = 0
                        # print(df['dată internare'][i])
            else:
                if math.isnan(df['dată internare'][i]):
                    df['dată internare'][i] = -1

            # codificare dată rezultat testare ca numarul zilei din an
            if isinstance(df['data rezultat testare'][i], datetime):
                # print(df['data rezultat testare'][i])
                df['data rezultat testare'][i] = int(df['data rezultat testare'][i].strftime("%j"))
            elif isinstance(df['data rezultat testare'][i], str):
                if 'NU' in df['data rezultat testare'][i].upper() or\
                        '-' in df['data rezultat testare'][i]:
                    df['data rezultat testare'][i] = -1
                else:
                    df['data rezultat testare'][i] = df['data rezultat testare'][i].replace(' ', '-')
                    df['data rezultat testare'][i] = df['data rezultat testare'][i].replace('.', '-')
                    df['data rezultat testare'][i] = df['data rezultat testare'][i].replace(',', '-')

                    if re.match(r"(\d+)-(\d+)-(\d\d\d\d)$", df['data rezultat testare'][i]):
                        df['data rezultat testare'][i] = datetime.strptime(df['data rezultat testare'][i], '%d-%m-%Y')
                        df['data rezultat testare'][i] = int(df['data rezultat testare'][i].strftime("%j"))
                    else:
                        # exceptii nerezolvate
                        # print(df['data rezultat testare'][i])
                        df['data rezultat testare'][i] = 0
            else:
                if math.isnan(df['data rezultat testare'][i]):
                    df['data rezultat testare'][i] = -1
            # print(df['instituția sursă'][i], df['sex'][i], df['vârstă'][i],
            # df['dată debut simptome declarate'][i], df['dată internare'][i], df['data rezultat testare'][i])

            # codificare istoric calatorii
            if isinstance(df['istoric de călătorie'][i], str):
                if re.match("( *)NU( *)", df['istoric de călătorie'][i].upper()) or \
                        re.match("( *)NEAG", df['istoric de călătorie'][i].upper()) or \
                        re.match("( *)FARA( *)", df['istoric de călătorie'][i].upper()) or \
                        re.match("( *)0( *)", df['istoric de călătorie'][i]):
                    df['istoric de călătorie'][i] = 0
                elif re.match("( *)DA( *)", df['istoric de călătorie'][i].upper()) or \
                        re.match("( *)1( *)", df['istoric de călătorie'][i]):
                    df['istoric de călătorie'][i] = 1
                else:
                    df['istoric de călătorie'][i] = 1
            elif math.isnan(df['istoric de călătorie'][i]):
                df['istoric de călătorie'][i] = 0
            elif df['istoric de călătorie'][i] == 0:
                df['istoric de călătorie'][i] = 0
            elif df['istoric de călătorie'][i] == 1:
                df['istoric de călătorie'][i] = 1

            # codificare mijloace de transport folosite
            if isinstance(df['mijloace de transport folosite'][i], str):
                if re.match("( *)NU( *)", df['mijloace de transport folosite'][i].upper()) or \
                        re.match("( *)NEAG", df['mijloace de transport folosite'][i].upper()) or \
                        re.match("( *)FARA( *)", df['mijloace de transport folosite'][i].upper()) or \
                        re.match("( *)0( *)", df['mijloace de transport folosite'][i]):
                    df['mijloace de transport folosite'][i] = 0
                elif re.match("( *)DA( *)", df['mijloace de transport folosite'][i].upper()) or \
                        re.match("( *)1( *)", df['mijloace de transport folosite'][i]):
                    df['mijloace de transport folosite'][i] = 1
                else:
                    df['mijloace de transport folosite'][i] = 1

            elif math.isnan(df['mijloace de transport folosite'][i]):
                df['mijloace de transport folosite'][i] = 0
            elif df['mijloace de transport folosite'][i] == 0:
                df['mijloace de transport folosite'][i] = 0
            elif df['mijloace de transport folosite'][i] == 1:
                df['mijloace de transport folosite'][i] = 1

            # codificare confirmare contact cu o persoană infectată
            if isinstance(df['confirmare contact cu o persoană infectată'][i], str):
                if re.match("( *)NU( *)", df['confirmare contact cu o persoană infectată'][i].upper()) or \
                        re.match("( *)NEAG", df['confirmare contact cu o persoană infectată'][i].upper()) or \
                        re.match("( *)FARA( *)", df['confirmare contact cu o persoană infectată'][i].upper()) or \
                        re.match("( *)0( *)", df['confirmare contact cu o persoană infectată'][i]):
                    df['confirmare contact cu o persoană infectată'][i] = 0
                elif re.match("( *)DA( *)", df['confirmare contact cu o persoană infectată'][i].upper()) or \
                        re.match("( *)1( *)", df['confirmare contact cu o persoană infectată'][i]):
                    df['confirmare contact cu o persoană infectată'][i] = 1
                else:
                    df['confirmare contact cu o persoană infectată'][i] = 1

            elif math.isnan(df['confirmare contact cu o persoană infectată'][i]):
                df['confirmare contact cu o persoană infectată'][i] = 0
            elif df['confirmare contact cu o persoană infectată'][i] == 0:
                df['confirmare contact cu o persoană infectată'][i] = 0
            elif df['confirmare contact cu o persoană infectată'][i] == 1:
                df['confirmare contact cu o persoană infectată'][i] = 1

        # verific ca toate valorile sa fie int-uri
        for i in df.index:
            if not isinstance(df['instituția sursă'][i], int) or\
                    not isinstance(df['sex'][i], int) or\
                    not isinstance(df['vârstă'][i], int) or\
                    not isinstance(df['dată debut simptome declarate'][i], int) or\
                    not isinstance(df['dată internare'][i], int) or\
                    not isinstance(df['data rezultat testare'][i], int) or\
                    not isinstance(df['istoric de călătorie'][i], int) or\
                    not isinstance(df['mijloace de transport folosite'][i], int) or\
                    not isinstance(df['confirmare contact cu o persoană infectată'][i], int):
                print(i, df['instituția sursă'][i], df['sex'][i], df['vârstă'][i],
                      df['dată debut simptome declarate'][i],
                      df['dată internare'][i], df['data rezultat testare'][i],
                      df['istoric de călătorie'][i],
                      df['mijloace de transport folosite'][i],
                      df['confirmare contact cu o persoană infectată'][i])
