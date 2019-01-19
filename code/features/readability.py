from nltk_contrib.readability.readabilitytests import ReadabilityTool
from sklearn.base import BaseEstimator
import numpy as np


__all__ = ['ReadabilityIndicesFeatures']


class ReadabilityIndicesFeatures(BaseEstimator):
    def get_feature_names(self):
        return np.array(
            ['GunningFOGIndex', 'FleschReadingEase', 'FleschKincaidGradeLevel', 'RIX', 'LIX', 'SMOGINDEX', 'ARI'])

    def fit(self, documents, y=None):
        return self

    def transform(self, documents):
        tool = ReadabilityTool()
        gunning_fog = [tool.GunningFogIndex(doc.content) for doc in documents]
        flesch_reading_ease = [tool.FleschReadingEase(doc.content) for doc in documents]
        rix = [tool.RIX(doc.content) for doc in documents]
        lix = [tool.LIX(doc.content) for doc in documents]
        smog_index = [tool.SMOGIndex(doc.content) for doc in documents]
        ari = [tool.ARI(doc.content) for doc in documents]
        flesch_kincaid_grade_level = [tool.FleschKincaidGradeLevel(doc.content) for doc in documents]

        X = np.array([gunning_fog, flesch_reading_ease, flesch_kincaid_grade_level, rix, lix, smog_index,ari]).T
        return X

    def fit_transform(self, documents, y=None):
        return self.transform(documents)

