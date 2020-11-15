# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math
import sys
from datetime import datetime
import pandas as pd
import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

np.set_printoptions(threshold=sys.maxsize)
pd.options.display.max_columns = None
pd.options.display.max_rows = None


def encode(file_name):
    df = pd.read_excel(file_name)
    df = pd.DataFrame(df, columns=df.columns)

    # replacing nan
    df['simptome declarate'] = df['simptome declarate'].replace(np.nan, 0)
    df['simptome raportate la internare'] = df['simptome raportate la internare'].replace(np.nan, 0)
    df['diagnostic și semne de internare'] = df['diagnostic și semne de internare'].replace(np.nan, 0)

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
                    df['vârstă'][i] = 0
        else:
            if math.isnan(df['vârstă'][i]):
                df['vârstă'][i] = -1

        # codificare dată debut simptome declarate ca numarul zilei din an
        if isinstance(df['dată debut simptome declarate'][i], datetime):
            # print(df['dată debut simptome declarate'][i])
            df['dată debut simptome declarate'][i] = int(df['dată debut simptome declarate'][i].strftime("%j"))
        elif isinstance(df['dată debut simptome declarate'][i], str):
            if 'NU' in df['dată debut simptome declarate'][i].upper() or \
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
            if 'NU' in df['dată internare'][i].upper() or \
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
            if 'NU' in df['data rezultat testare'][i].upper() or \
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

        # codificare simptome
        if isinstance(df['simptome declarate'][i], str):
            if re.match("( *)ASIMPTOMATIC( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)NU( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)FARA( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)ABSENTE( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)AFEBRIL( *)", df['simptome declarate'][i].upper()):
                df['simptome declarate'][i] = 0
            elif re.match("( *)TUS( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)DIAREE( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)OBOSEALA( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)DISP( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)RESPIRAT( *)", df['simptome declarate'][i].upper()) or \
                    re.match("( *)FEBRA( *)", df['simptome declarate'][i]):
                df['simptome declarate'][i] = 1
            else:
                df['simptome declarate'][i] = -1

        # codificare simptome
        if isinstance(df['simptome raportate la internare'][i], str):
            if re.match("( *)ASIM( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)NU( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)FARA( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)ABSENTE( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)AFEBRIL( *)", df['simptome raportate la internare'][i].upper()):
                df['simptome raportate la internare'][i] = 0
            elif re.match("( *)TUS( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)DIAREE( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)OBOSEALA( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)DISP( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)INSUF( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)PIERDERE( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)RESPIRAT( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)FEBRA( *)", df['simptome raportate la internare'][i]):
                df['simptome raportate la internare'][i] = 1
            else:
                df['simptome raportate la internare'][i] = -1


        # codificare simptome
        if isinstance(df['simptome raportate la internare'][i], str):
            if re.match("( *)ASIM( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)NU( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)FARA( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)ABSENTE( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)AFEBRIL( *)", df['simptome raportate la internare'][i].upper()):
                df['simptome raportate la internare'][i] = 0
            elif re.match("( *)TUS( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)DIAREE( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)OBOSEALA( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)DISP( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)INSUF( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)PIERDERE( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)RESP( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)COV( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)SARS( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)SUSP( *)", df['simptome raportate la internare'][i].upper()) or \
                    re.match("( *)FEBRA( *)", df['simptome raportate la internare'][i]):
                df['simptome raportate la internare'][i] = 1
            else:
                df['simptome raportate la internare'][i] = -1

    # verific ca toate valorile sa fie int-uri
    for i in df.index:
        if not isinstance(df['instituția sursă'][i], int) or \
                not isinstance(df['sex'][i], int) or \
                not isinstance(df['vârstă'][i], int) or \
                not isinstance(df['dată debut simptome declarate'][i], int) or \
                not isinstance(df['dată internare'][i], int) or \
                not isinstance(df['data rezultat testare'][i], int) or \
                not isinstance(df['istoric de călătorie'][i], int) or \
                not isinstance(df['mijloace de transport folosite'][i], int) or \
                not isinstance(df['confirmare contact cu o persoană infectată'][i], int):
            print(i, df['instituția sursă'][i], df['sex'][i], df['vârstă'][i],
                  df['dată debut simptome declarate'][i],
                  df['dată internare'][i], df['data rezultat testare'][i],
                  df['istoric de călătorie'][i],
                  df['mijloace de transport folosite'][i],
                  df['confirmare contact cu o persoană infectată'][i])

    return 'Uneori este greu, alteori nu este usor.'
