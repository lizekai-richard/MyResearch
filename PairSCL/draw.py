#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim: tabstop=4 shiftwidth=4 expandtab number

"""
Author: Wesley (liwanshui12138@gmail.com)
Date: 2022-8-2
"""
import numpy as np
import os
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

MODEL = '256'
fig_dir = os.path.join('fig', f'SKILL_BERT_{MODEL}_lr_5e-05_decay_0.0001_bsz_32_temp_0.05')
os.makedirs(fig_dir, exist_ok=True)

x_train = np.load(f'save/SKILL_models/SKILL_BERT_{MODEL}_lr_5e-05_decay_0.0001_bsz_32_temp_0.05/train_input.npy')  # (8548,256)
y_train = np.load(f'save/SKILL_models/SKILL_BERT_{MODEL}_lr_5e-05_decay_0.0001_bsz_32_temp_0.05/train_label.npy')  # (8548,)
x_tr_proj = np.load(f'save/SKILL_models/SKILL_BERT_{MODEL}_lr_5e-05_decay_0.0001_bsz_32_temp_0.05/train_proj.npy')  # (8548,768)
x_test = np.load(f'save/SKILL_models/SKILL_BERT_{MODEL}_lr_5e-05_decay_0.0001_bsz_32_temp_0.05/test_input.npy')  # (1007,256)
y_test = np.load(f'save/SKILL_models/SKILL_BERT_{MODEL}_lr_5e-05_decay_0.0001_bsz_32_temp_0.05/test_label.npy')  # (1007,)
x_te_proj = np.load(f'save/SKILL_models/SKILL_BERT_{MODEL}_lr_5e-05_decay_0.0001_bsz_32_temp_0.05/test_proj.npy')  # (1007,768)
print(x_train.shape, x_test.shape)

# No PCA on proj
x_te_proj_df = pd.DataFrame(x_te_proj[:, :2], columns=['Proj1', 'Proj2'])  # (10000, 256) -> (10000, 2)
x_te_proj_df['label'] = y_test
fig, ax = plt.subplots()
ax = sns.scatterplot('Proj1', 'Proj2', data=x_te_proj_df, palette='tab10', hue='label', linewidth=0, alpha=0.6, ax=ax)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
title = f"Data_SKILL_Embedding_contrastive_PCA_False_{MODEL}"
ax.set_title(title)
fig.savefig(os.path.join(fig_dir, title))
g = sns.jointplot('Proj1', 'Proj2', data=x_te_proj_df, kind="hex")
plt.subplots_adjust(top=0.95)
title = 'Joint_Plot_' + title
g.fig.suptitle(title)
g.savefig(os.path.join(fig_dir, title))
plt.show()


# do PCA for the projected data
pca = PCA(n_components=2)
pca.fit(x_tr_proj)
x_te_proj_pca = pca.transform(x_te_proj)  # (10000, 2)
# draw
x_te_proj_pca_df = pd.DataFrame(x_te_proj_pca, columns=['PC1', 'PC2'])
x_te_proj_pca_df['label'] = y_test
fig, ax = plt.subplots()
ax = sns.scatterplot('PC1', 'PC2', data=x_te_proj_pca_df, palette='tab10', hue='label', linewidth=0, alpha=0.6, ax=ax)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
title = f"Data_SKILL_Embedding_contrastive_PCA_True_{MODEL}"
ax.set_title(title)
fig.savefig(os.path.join(fig_dir, title))
g = sns.jointplot('PC1', 'PC2', data=x_te_proj_pca_df, kind="hex")
plt.subplots_adjust(top=0.95)
title = 'Joint_Plot_' + title
g.fig.suptitle(title)
g.savefig(os.path.join(fig_dir, title))
plt.show()


# do PCA for original data
pca = PCA(n_components=2)
pca.fit(x_train)
x_te_pca = pca.transform(x_test)  # (10000, 2)
# draw
x_te_pca_df = pd.DataFrame(x_te_pca, columns=['PC1', 'PC2'])
x_te_pca_df['label'] = y_test
fig, ax = plt.subplots()
ax = sns.scatterplot('PC1', 'PC2', data=x_te_pca_df, palette='tab10', hue='label', linewidth=0, alpha=0.6, ax=ax)
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
title = f"Data_SKILL_Embedding_none_PCA_True_{MODEL}"
ax.set_title(title)
fig.savefig(os.path.join(fig_dir, title))
g = sns.jointplot('PC1', 'PC2', data=x_te_pca_df, kind="hex")
plt.subplots_adjust(top=0.95)
title = 'Joint_Plot_' + title
g.fig.suptitle(title)
g.savefig(os.path.join(fig_dir, title))
plt.show()

