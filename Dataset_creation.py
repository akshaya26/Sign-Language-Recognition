import splitfolders
"""
Dataset link : https://www.kaggle.com/datasets/vaishnaviasonawane/indian-sign-language-dataset
"""
splitfolders.ratio("Indian_Split", output="Indian", seed=12, ratio=(.8, 0.0, 0.2))


