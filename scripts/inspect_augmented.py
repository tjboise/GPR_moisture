"""Quick inspection of saved augmented data."""
import numpy as np
import pandas as pd
import pathlib

for method in ['gan', 'noise']:
    print(f'\n{"="*50}')
    print(f'  {method.upper()} augmented data')
    print(f'{"="*50}')
    for fold in range(1, 6):
        fold_dir = pathlib.Path(f'../data/augmented/{method}/fold_{fold}')
        X = np.load(fold_dir / 'X_synth.npy')
        y = np.load(fold_dir / 'y_synth.npy')
        meta = pd.read_csv(fold_dir / 'meta.csv', index_col=0)

        print(f'\nFold {fold}:')
        print(f'  X_synth shape : {X.shape}  (samples × time points)')
        print(f'  X value range : {X.min():.3f} ~ {X.max():.3f}')
        print(f'  y moisture (%):\n{pd.DataFrame(y, columns=["S","T","M","B"]).describe().round(2).to_string()}')
        print(f'  meta preview  :\n{meta.head(3).to_string()}')
