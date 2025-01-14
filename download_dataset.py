#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import kagglehub # type: ignore

# Download latest version
path = kagglehub.dataset_download("schran/insurance-premium-prediction")

print("Path to dataset files:", path)

exec(open('supervised_learning_insurance.py').read())


# # *Link to dataset*

# https://www.kaggle.com/datasets/schran/insurance-premium-prediction/data

# # *Metadata*

# # Collaborators
# Saravanan G (Owner)
# 
# # License
# Apache 2.0
# 
# # Expected Update Frequency
# Not specified (Updated 4 months ago)

# %%
