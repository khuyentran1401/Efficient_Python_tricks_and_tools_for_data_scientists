from pyinstrument import Profiler
import pandas as pd
import numpy as np

df = pd.DataFrame({'nums': np.random.randint(0, 100, 10000)})
def is_even(num: int) -> int:
    return num % 2 == 0

profiler = Profiler()
profiler.start()

df = df.assign(is_even=lambda df_: is_even(df_.nums))

profiler.stop()
profiler.print()
