import pandas as pd
import openpyxl

df = pd.DataFrame([["name", "abed", "ibrahem"], ["time", 22, 32]],
                  index=['1', '2'], columns=['a', 'b', 'c'])

print(df)

df.to_excel('data.xlsx', index=False, header=False)