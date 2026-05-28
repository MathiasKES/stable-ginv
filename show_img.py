import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from functions.Dataset import load_dataset

dst, _, _, _ = load_dataset('cifar100', '/work3/s234843/bachelor/datasets')
img, label = dst[23784]
fig, ax = plt.subplots()
ax.imshow(np.array(img))
ax.set_title(f'idx=23784  label={label}')
ax.axis('off')
fig.savefig('idx_23784.png', dpi=150, bbox_inches='tight')
print(f'Saved idx_23784.png  label={label}')
