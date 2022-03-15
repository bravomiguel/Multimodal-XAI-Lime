# -*- coding: utf-8 -*-
"""
Custom version of LIME module, adapted to deal with multi-input model used for coursework.
"""
import numpy as np
import scipy as sp
from scipy.stats.distributions import norm

from lime.lime_image import LimeImageExplainer, ImageExplanation
from lime.lime_text import LimeTextExplainer, IndexedCharacters, IndexedString, TextDomainMapper
from lime.lime_tabular import TableDomainMapper, LimeTabularExplainer
from lime.wrappers.scikit_image import SegmentationAlgorithm
from lime import explanation

import sklearn
import sklearn.preprocessing

from skimage.color import gray2rgb
from tqdm.auto import tqdm
import copy
import warnings
from pyDOE2 import lhs


class CustomLimeImageExplainer(LimeImageExplainer):
    def __init__(self, kernel_width=.25, kernel=None, verbose=False,
                 feature_selection='auto', random_state=None):
        super().__init__(kernel_width=kernel_width, kernel=kernel, verbose=verbose,
                 feature_selection=feature_selection, random_state=random_state)
    
    def explain_instance(self, image, pet_id, classifier_fn, labels=(1,),
                         hide_color=None,
                         top_labels=5, num_features=100000, num_samples=1000,
                         batch_size=10,
                         segmentation_fn=None,
                         distance_metric='cosine',
                         model_regressor=None,
                         random_seed=None,
                         progress_bar=True):
     
        # turn image to rgb if grayscale
        if len(image.shape) == 2:
            image = gray2rgb(image)
        if random_seed is None:
            random_seed = self.random_state.randint(0, high=1000)
        
        # initiate segmentation algorithm
        if segmentation_fn is None:
            segmentation_fn = SegmentationAlgorithm('quickshift', kernel_size=4,
                                                    max_dist=200, ratio=0.2,
                                                    random_seed=random_seed)
        # segment image
        try:
            segments = segmentation_fn(image)
        except ValueError as e:
            raise e

        # decide how to fudge image segments (average segment color or same color throughout)
        fudged_image = image.copy()
        if hide_color is None:
            # average color of each segment
            for x in np.unique(segments):
                fudged_image[segments == x] = (
                    np.mean(image[segments == x][:, 0]),
                    np.mean(image[segments == x][:, 1]),
                    np.mean(image[segments == x][:, 2]))
        else:
            # same color throughout
            fudged_image[:] = hide_color

        top = labels
        
        # get data and labels to train linear classifier
        data, labels = self.data_labels(image, 
                                        pet_id, 
                                        fudged_image, 
                                        segments,
                                        classifier_fn, 
                                        num_samples,
                                        batch_size=batch_size,
                                        progress_bar=progress_bar)

        # calculate distances of each sample to original sample 
        distances = sklearn.metrics.pairwise_distances(
            data,
            data[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()
            
        # explain original image
        ret_exp = ImageExplanation(image, segments)
        if top_labels:
            # get indices of top labels for original image 
            top = np.argsort(labels[0])[-top_labels:]
            # set top label indices in explanation
            ret_exp.top_labels = list(top)
            # reverse order of top label indices
            ret_exp.top_labels.reverse()
            
        for label in top:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, labels, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def data_labels(self,
                    image,
                    pet_id,
                    fudged_image,
                    segments,
                    classifier_fn,
                    num_samples,
                    batch_size=10,
                    progress_bar=True):
   
        # get number of features for linear classifier (i.e. image segments)
        n_features = np.unique(segments).shape[0]
        # random array of 0's and 1's, indicating features (segments) to leave on for each sample
        data = self.random_state.randint(0, 2, num_samples * n_features)\
            .reshape((num_samples, n_features))
        labels = []
        # make sure first sample has all segments switched on (i.e. original image)
        data[0, :] = 1
        imgs = []
        # switch on progress bar for iterating over rows in data (i.e. samples)
        rows = tqdm(data) if progress_bar else data
        # iterate over samples/rows
        for row in rows:
            # make copy of image
            temp = copy.deepcopy(image)
            # get indices where segment is 0 (i.e. switched off segments)
            zeros = np.where(row == 0)[0]
            # create mask of False values for image
            mask = np.zeros(segments.shape).astype(bool)
            # set mask to True for switched off segments
            for z in zeros:
                mask[segments == z] = True
            # use mask to fudge (i.e. switch off) relevant segments from image
            temp[mask] = fudged_image[mask]
            # save amended images (i.e. data) into batch (i.e. list)
            imgs.append(temp)
            # when image batch gets to 10, run classifier on batch, and get corresponding preds (labels for linear classifier)
            if len(imgs) == batch_size:
                # run classifier
                preds = classifier_fn(np.array(imgs), pet_id)
                # save labels
                labels.extend(preds)
                # reset batch
                imgs = []
        # run classifier on last batch
        if len(imgs) > 0:
            # run classifier
            preds = classifier_fn(np.array(imgs), pet_id)
            # save labels
            labels.extend(preds)
        return data, np.array(labels)
    
class CustomLimeTextExplainer(LimeTextExplainer):
    def __init__(self,
                 kernel_width=25,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 split_expression=r'\W+',
                 bow=True,
                 mask_string=None,
                 random_state=None,
                 char_level=False):
        super().__init__(kernel_width=kernel_width,
                         kernel=kernel,
                         verbose=verbose,
                         class_names=class_names,
                         feature_selection=feature_selection,
                         split_expression=split_expression,
                         bow=bow,
                         mask_string=mask_string,
                         random_state=random_state,
                         char_level=char_level)

    def explain_instance(self,
                         text_instance,
                         classifier_fn,
                         pet_id,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='cosine',
                         model_regressor=None):

        # index characters in text
        indexed_string = (IndexedCharacters(
            text_instance, bow=self.bow, mask_string=self.mask_string)
                          if self.char_level else
                          IndexedString(text_instance, bow=self.bow,
                                        split_expression=self.split_expression,
                                        mask_string=self.mask_string))
        domain_mapper = TextDomainMapper(indexed_string)
        data, yss, distances = self.__data_labels_distances(
            indexed_string, classifier_fn, pet_id, num_samples,
            distance_metric=distance_metric)
        if self.class_names is None:
            self.class_names = [str(x) for x in range(yss[0].shape[0])]
        ret_exp = explanation.Explanation(domain_mapper=domain_mapper,
                                          class_names=self.class_names,
                                          random_state=self.random_state)
        ret_exp.predict_proba = yss[0]
        if top_labels:
            labels = np.argsort(yss[0])[-top_labels:]
            ret_exp.top_labels = list(labels)
            ret_exp.top_labels.reverse()
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                data, yss, distances, label, num_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp

    def __data_labels_distances(self,
                                indexed_string,
                                classifier_fn,
                                pet_id,
                                num_samples,
                                distance_metric='cosine'):

        def distance_fn(x):
            return sklearn.metrics.pairwise.pairwise_distances(
                x, x[0], metric=distance_metric).ravel() * 100

        # get number of distinct words in text
        doc_size = indexed_string.num_words()
        # create array with random number of words to sample from doc each time
        sample = np.random.randint(1, doc_size + 1, num_samples - 1)
        # create dataset with 1's - template for then removing words from each sample
        data = np.ones((num_samples, doc_size))
        # set first sample to be original text
        data[0] = np.ones(doc_size)
        # create list of indices for each word
        features_range = range(doc_size)
        # get raw text in list
        inverse_data = [indexed_string.raw_string()]
        # create samples with words removed
        for i, size in enumerate(sample, start=1):
            # create index mask to sample words to remove from doc each time
            inactive = np.random.choice(features_range, size, replace=False)
            # use index mask to 'switch off' words for removal from data array
            data[i, inactive] = 0
            inverse_data.append(indexed_string.inverse_removing(inactive))
        # get preds (labels) to train linear classifier
        labels = classifier_fn(inverse_data, pet_id)
        # get distances of samples from original
        distances = distance_fn(sp.sparse.csr_matrix(data))
        return data, labels, distances
    
    
class CustomLimeTabularExplainer(LimeTabularExplainer):

    def __init__(self, 
                 training_data,
                 mode="classification",
                 training_labels=None,
                 feature_names=None,
                 categorical_features=None,
                 categorical_names=None,
                 kernel_width=None,
                 kernel=None,
                 verbose=False,
                 class_names=None,
                 feature_selection='auto',
                 discretize_continuous=True,
                 discretizer='quartile',
                 sample_around_instance=False,
                 random_state=None,
                 training_data_stats=None):
        super().__init__(training_data,
                         mode=mode,
                         training_labels=training_labels,
                         feature_names=feature_names,
                         categorical_features=categorical_features,
                         categorical_names=categorical_names,
                         kernel_width=kernel_width,
                         kernel=kernel,
                         verbose=verbose,
                         class_names=class_names,
                         feature_selection=feature_selection,
                         discretize_continuous=discretize_continuous,
                         discretizer=discretizer,
                         sample_around_instance=sample_around_instance,
                         random_state=random_state,
                         training_data_stats=training_data_stats)

    def explain_instance(self,
                         data_row,
                         predict_fn,
                         pet_id,
                         labels=(1,),
                         top_labels=None,
                         num_features=10,
                         num_samples=5000,
                         distance_metric='euclidean',
                         model_regressor=None,
                         sampling_method='gaussian'):

        if sp.sparse.issparse(data_row) and not sp.sparse.isspmatrix_csr(data_row):
            # Preventative code: if sparse, convert to csr format if not in csr format already
            data_row = data_row.tocsr()
        data, inverse = self.__data_inverse(data_row, num_samples, sampling_method)
        if sp.sparse.issparse(data):
            # Note in sparse case we don't subtract mean since data would become dense
            scaled_data = data.multiply(self.scaler.scale_)
            # Multiplying with csr matrix can return a coo sparse matrix
            if not sp.sparse.isspmatrix_csr(scaled_data):
                scaled_data = scaled_data.tocsr()
        else:
            scaled_data = (data - self.scaler.mean_) / self.scaler.scale_
        distances = sklearn.metrics.pairwise_distances(
                scaled_data,
                scaled_data[0].reshape(1, -1),
                metric=distance_metric
        ).ravel()

        yss = predict_fn(inverse, pet_id)

        # for classification, the model needs to provide a list of tuples - classes
        # along with prediction probabilities
        if self.mode == "classification":
            if len(yss.shape) == 1:
                raise NotImplementedError("LIME does not currently support "
                                          "classifier models without probability "
                                          "scores. If this conflicts with your "
                                          "use case, please let us know: "
                                          "https://github.com/datascienceinc/lime/issues/16")
            elif len(yss.shape) == 2:
                if self.class_names is None:
                    self.class_names = [str(x) for x in range(yss[0].shape[0])]
                else:
                    self.class_names = list(self.class_names)
                if not np.allclose(yss.sum(axis=1), 1.0):
                    warnings.warn("""
                    Prediction probabilties do not sum to 1, and
                    thus does not constitute a probability space.
                    Check that you classifier outputs probabilities
                    (Not log probabilities, or actual class predictions).
                    """)
            else:
                raise ValueError("Your model outputs "
                                 "arrays with {} dimensions".format(len(yss.shape)))

        # for regression, the output should be a one-dimensional array of predictions
        else:
            try:
                if len(yss.shape) != 1 and len(yss[0].shape) == 1:
                    yss = np.array([v[0] for v in yss])
                assert isinstance(yss, np.ndarray) and len(yss.shape) == 1
            except AssertionError:
                raise ValueError("Your model needs to output single-dimensional \
                    numpyarrays, not arrays of {} dimensions".format(yss.shape))

            predicted_value = yss[0]
            min_y = min(yss)
            max_y = max(yss)

            # add a dimension to be compatible with downstream machinery
            yss = yss[:, np.newaxis]

        feature_names = copy.deepcopy(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(data_row.shape[0])]

        if sp.sparse.issparse(data_row):
            values = self.convert_and_round(data_row.data)
            feature_indexes = data_row.indices
        else:
            values = self.convert_and_round(data_row)
            feature_indexes = None

        for i in self.categorical_features:
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = int(data_row[i])
            if i in self.categorical_names:
                name = self.categorical_names[i][name]
            feature_names[i] = '%s = %s' % (feature_names[i], name)
            values[i] = 'True'
        categorical_features = self.categorical_features
        # print(feature_names)

        discretized_feature_names = None
        if self.discretizer is not None:
            categorical_features = range(data.shape[1])
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = copy.deepcopy(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                        discretized_instance[f])]

        domain_mapper = TableDomainMapper(feature_names,
                                          values,
                                          scaled_data[0],
                                          categorical_features=categorical_features,
                                          discretized_feature_names=discretized_feature_names,
                                          feature_indexes=feature_indexes)
        ret_exp = explanation.Explanation(domain_mapper,
                                          mode=self.mode,
                                          class_names=self.class_names)
        if self.mode == "classification":
            ret_exp.predict_proba = yss[0]
            if top_labels:
                labels = np.argsort(yss[0])[-top_labels:]
                ret_exp.top_labels = list(labels)
                ret_exp.top_labels.reverse()
        else:
            ret_exp.predicted_value = predicted_value
            ret_exp.min_value = min_y
            ret_exp.max_value = max_y
            labels = [0]
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score, ret_exp.local_pred) = self.base.explain_instance_with_data(
                    scaled_data,
                    yss,
                    distances,
                    label,
                    num_features,
                    model_regressor=model_regressor,
                    feature_selection=self.feature_selection)

        if self.mode == "regression":
            ret_exp.intercept[1] = ret_exp.intercept[0]
            ret_exp.local_exp[1] = [x for x in ret_exp.local_exp[0]]
            ret_exp.local_exp[0] = [(i, -1 * j) for i, j in ret_exp.local_exp[1]]

        return ret_exp

    def __data_inverse(self,
                       data_row,
                       num_samples,
                       sampling_method):
        is_sparse = sp.sparse.issparse(data_row)
        if is_sparse:
            num_cols = data_row.shape[1]
            data = sp.sparse.csr_matrix((num_samples, num_cols), dtype=data_row.dtype)
        else:
            num_cols = data_row.shape[0]
            data = np.zeros((num_samples, num_cols))
        categorical_features = range(num_cols)
        if self.discretizer is None:
            instance_sample = data_row
            scale = self.scaler.scale_
            mean = self.scaler.mean_
            if is_sparse:
                # Perturb only the non-zero values
                non_zero_indexes = data_row.nonzero()[1]
                num_cols = len(non_zero_indexes)
                instance_sample = data_row[:, non_zero_indexes]
                scale = scale[non_zero_indexes]
                mean = mean[non_zero_indexes]

            if sampling_method == 'gaussian':
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)
            elif sampling_method == 'lhs':
                data = lhs(num_cols, samples=num_samples
                           ).reshape(num_samples, num_cols)
                means = np.zeros(num_cols)
                stdvs = np.array([1]*num_cols)
                for i in range(num_cols):
                    data[:, i] = norm(loc=means[i], scale=stdvs[i]).ppf(data[:, i])
                data = np.array(data)
            else:
                warnings.warn('''Invalid input for sampling_method.
                                 Defaulting to Gaussian sampling.''', UserWarning)
                data = self.random_state.normal(0, 1, num_samples * num_cols
                                                ).reshape(num_samples, num_cols)
                data = np.array(data)

            if self.sample_around_instance:
                data = data * scale + instance_sample
            else:
                data = data * scale + mean
            if is_sparse:
                if num_cols == 0:
                    data = sp.sparse.csr_matrix((num_samples,
                                                 data_row.shape[1]),
                                                dtype=data_row.dtype)
                else:
                    indexes = np.tile(non_zero_indexes, num_samples)
                    indptr = np.array(
                        range(0, len(non_zero_indexes) * (num_samples + 1),
                              len(non_zero_indexes)))
                    data_1d_shape = data.shape[0] * data.shape[1]
                    data_1d = data.reshape(data_1d_shape)
                    data = sp.sparse.csr_matrix(
                        (data_1d, indexes, indptr),
                        shape=(num_samples, data_row.shape[1]))
            categorical_features = self.categorical_features
            first_row = data_row
        else:
            first_row = self.discretizer.discretize(data_row)
        data[0] = data_row.copy()
        inverse = data.copy()
        for column in categorical_features:
            values = self.feature_values[column]
            freqs = self.feature_frequencies[column]
            inverse_column = self.random_state.choice(values, size=num_samples,
                                                      replace=True, p=freqs)
            binary_column = (inverse_column == first_row[column]).astype(int)
            binary_column[0] = 1
            inverse_column[0] = data[0, column]
            data[:, column] = binary_column
            inverse[:, column] = inverse_column
        if self.discretizer is not None:
            inverse[1:] = self.discretizer.undiscretize(inverse[1:])
        inverse[0] = data_row
        return data, inverse

