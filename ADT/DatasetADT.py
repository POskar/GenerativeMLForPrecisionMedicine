import copy
import logging
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.INFO)

def one_hot_encode(x):
    """
    Transform a binary label into a one-hot encoding.
    """
    if x == 0.:
        return np.array([1., 0.])
    elif x == 1.:
        return np.array([0., 1.])
    else:
        raise ValueError("Label can be only 0. or 1.")

class Scaler:
    """
    Scaler for clinical and imaging data.

    Clinical data can be standardized to zero mean and unit variance or min-max scaled between 0 and 1.
    Imaging data is always standardized to zero mean and unit variance.

    Standardization: Calculates and stores the mean and std of the given data and allows to scale
    each batch of the loaded data.

    Min-max scaling: Data without missing values is scaled to the range [0, 1]. The transformation is given by:
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min
        If data_max is given, the x_max is not calculated from the data but taken from the given dictionary.
        Higher values than given in data_max are clipped to the max_value given in data_max.
        If data_min is given, the x_min is not calculated from the data but taken from the given dictionary.
        Lower values than given in data_min are clipped to the min_value given in data_min.
        Missing values are set to -1.
    """

    def __init__(self, numerical_data: list, scaler_type: str, data_max: list = None, data_min: list = None,
                 missing_values: list = None):
        """
        Args:
            numerical_data (list): list of numerical feature names from the clinical data.
            scaler_type (str): type of scaler for clinical data. Imaging data is always standardized.
                                Options:
                                - 'standard' for zero mean and unit variance standardization
                                - 'minmax' for min-max scaling
            data_max (list): OPTIONAL list with pre-defined max values for each numerical feature in the same order as
                             numerical_data.
                             Only applies to 'minmax'. If not given, the max values are calculated from the data.
            data_min (list): OPTIONAL list with pre-defined min values for each numerical feature in the same order as
                             numerical_data.
                             Only applies to 'minmax'. If not given, the min values are calculated from the data.
            missing_values (int): List of values that is used to represent missing values in the input data
                                  in the same order as numerical_data.
                                  MANDATORY to for min-max scaling.

        """
        self.logger = logging.getLogger('Scaler')
        self.logger.setLevel('DEBUG')
        self.STANDARD = "standard"
        self.MINMAX = "minmax"

        self.numerical_data = numerical_data
        self.scaler_type = scaler_type
        self.logger.debug(f'scaler_type: {self.scaler_type}')

        # statistics for clinical data
        if self.scaler_type == self.STANDARD:
            self.clin_mean_list = []
            self.clin_std_list = []
        elif self.scaler_type == self.MINMAX:
            self.data_min = data_min
            self.data_max = data_max
            self.missing_values = missing_values

            if not self.missing_values:
                raise AttributeError('missing_values must be given for min-max scaling.')
            if len(self.missing_values) != len(self.numerical_data):
                raise AssertionError('missing_values must have the same length as numerical_data.')
        else:
            raise ValueError(f'Invalid scaler_type. Scaler_type must be either {self.STANDARD} or {self.MINMAX}')

        # statistics for imaging data
        self.img_mean_dict = {}
        self.img_std_dict = {}
        self.merged_img_mean_list = []
        self.merged_img_std_list = []


    def preprocess_clinical_data(self, batch):
        """
        Scale the clinical data.

        Args:
            batch: batch of clinical data.
        """
        # check inputs
        if not isinstance(batch, np.ndarray):
            raise TypeError('Batch must be a np.ndarray.')

        if self.scaler_type == self.STANDARD:
            return self.standardize_clinical_data(batch)
        if self.scaler_type == self.MINMAX:
            return self.minmax_scale_clinical_data(batch)
        raise ValueError(f'scaler_type must be either {self.STANDARD} or {self.MINMAX}')

    def standardize_clinical_data(self, batch):
        """
        Standardize the clinical data.

        Args:
            batch: batch of clinical data.
        """
        if len(self.clin_mean_list) == 0 or len(self.clin_std_list) == 0:
            raise AssertionError(
                "The scaler was not fit on clinical data yet. Use the fit method before preprocessing data.")

        batch = copy.deepcopy(batch)
        for i, ind in enumerate(self.numerical_data_indices):
            batch[:, ind] = (batch[:, ind] - self.clin_mean_list[i]) / self.clin_std_list[i]

        return batch

    def minmax_scale_clinical_data(self, batch):
        """
        Minmax scale the clinical data.

        Args:
            batch: 2D batch of clinical data.

        Returns:
            numpy.ndarray: Scaled batch of clinical data.
        """
        # Check if batch is a 2D numpy array
        if len(batch.shape) != 2:
            raise ValueError("Input 'batch' must be a 2D numpy array.")

        batch = copy.deepcopy(batch)

        # Ensure data_max and missing_values are defined
        if any(array is None or len(array) == 0 for array in [self.data_max, self.data_min, self.missing_values]):
            raise AssertionError("The scaler was not fit on clinical data yet. "
                                 "data_max, data_min and missing_values must be defined for min-max scaling.")

        # Convert to pandas dataframe, so that we can use np.nan. Ndarray does not support nan.
        scaled_df = pd.DataFrame(batch)
        for col_idx, missing_value, minimum, maximum in zip(self.numerical_data_indices, self.missing_values,
                                                            self.data_min, self.data_max):
            # Replace missing values with numpy.nan
            scaled_df.iloc[:, col_idx] = scaled_df.iloc[:, col_idx].replace(missing_value, np.nan)

            # Clip values to maximum
            scaled_df.iloc[:, col_idx] = scaled_df.iloc[:, col_idx].clip(lower=minimum, upper=maximum)

            # Min-max scale each column
            scaled_df.iloc[:, col_idx] = (scaled_df.iloc[:, col_idx] - minimum) / (maximum - minimum)

        # Set missing values to -1
        scaled_df = scaled_df.fillna(-1)
        self.logger.debug(f'scaled_df head: {scaled_df.head(5)}')

        # Convert back to numpy array
        return scaled_df.to_numpy()

    def preprocess_merged_data(self, batch):
        """
        Standardize the merged imaging data.

        Args:
            batch: batch of merged imaging data.
        """
        if not isinstance(batch, np.ndarray):
            raise TypeError('Batch must be a np.ndarray.')
        if len(self.merged_img_mean_list) == 0 and len(self.merged_img_std_list) == 0:
            raise AssertionError(
                "The scaler was not fit on merged imaging data yet. Use the fit method before preprocessing data.")
        if len(self.merged_img_mean_list) != batch.shape[-1]:
            raise AssertionError("The number of channels in merged imaging data must be equal to the number of "
                                 "means in img_mean_list in Scaler. "
                                 "The number of channels in merged imaging data is " + str(batch.shape[-1]) + ". "
                                 "The batch shape is " + str(batch.shape) + ". "                                                                             
                                 "The number of means in img_mean_list is " + str(len(self.merged_img_mean_list)))

        batch = copy.deepcopy(batch)

        i = 0
        for mean, std in zip(self.merged_img_mean_list, self.merged_img_std_list):
            batch[..., i] = (batch[..., i] - mean) / std
            i += 1

        return batch

