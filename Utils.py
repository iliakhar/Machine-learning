import pandas as pd
import numpy as np



def GetDataFrame(filename: str):
    dataFrame: pd.DataFrame = pd.read_csv(filename)

    colTypes = dict.fromkeys(dataFrame.columns, np.int64)

    for col in dataFrame.columns:
        if dataFrame[col].dtype == np.object_:
            dataFrame[col] = dataFrame[col].str.replace('?', '-1')
    dataFrame = dataFrame.astype(colTypes)
    return dataFrame


def get_data_frame(filename: str):
    df: pd.DataFrame = pd.read_csv(filename, header=0).fillna(-1)
    df.loc[df.type == 'white', 'type'] = np.float64(0) #разделение данных по цвету
    df.loc[df.type == 'red', 'type'] = np.float64(1)
    # print(df)
    df.info()

    return df
