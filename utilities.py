import numpy as np
from sklearn.base import TransformerMixin
import mne
import random as rnd


###### Sliding window average on data
def sliding_window(epoch,sliding_window_size=10, sliding_window_step=1,sliding_window_min_size=None):
    """
    This function outputs an epoch object that has been built from a sliding window on the data

    :param epoch:
    :param sliding_window_size: Window size (number of data points) over which the data is averaged
    :param sliding_window_step: Step in number of data points. 4 corresponds to 10 ms.
    :param sliding_window_min_size: The last window minimal size.
    :return:
    """

    xformer =SlidingWindow(window_size=sliding_window_size, step=sliding_window_step,
                                         min_window_size=sliding_window_min_size)

    n_time_points = epoch._data.shape[2]
    window_start = np.array(range(0, n_time_points - sliding_window_size + 1, sliding_window_step))
    window_end = window_start + sliding_window_size

    window_end[-1] = min(window_end[-1], n_time_points)  # make sure that the last window doesn't exceed the input size

    intermediate_times = [int((window_start[i] + window_end[i]) / 2) for i in range(len(window_start))]
    times = epoch.times[intermediate_times]

    epoch2 = mne.EpochsArray(xformer.fit_transform(epoch._data),epoch.info)
    epoch2._set_times(times)
    epoch2.metadata = epoch.metadata

    return epoch2

class SlidingWindow(TransformerMixin):
    """
    Aggregate time points in a "sliding window" manner

    Input: Anything x Anything x Time points
    Output - if averaging: Unchanged x Unchanged x Windows
    Output - if not averaging: Windows x Unchanged x Unchanged x Window size
                Note that in this case, the output may not be a real matrix in case the last sliding window is smaller than the others
    """

    #--------------------------------------------------
    def __init__(self, window_size, step, min_window_size=None, average=True, debug=False):
        """
        :param window_size: The no. of time points to average
        :param step: The no. of time points to slide the window to get the next result
        :param min_window_size: The minimal number of time points acceptable in the last step of the sliding window.
                                If None: min_window_size will be the same as window_size
        :param average: If True, just reduce the number of time points by averaging over each window
                        If False, each window is copied as-is to the output, without averaging
        """
        self._window_size = window_size
        self._step = step
        self._min_window_size = min_window_size
        self._average = average
        self._debug = debug


    #--------------------------------------------------
    # noinspection PyUnusedLocal
    def fit(self, x, y=None, *_):
        return self

    #--------------------------------------------------
    def transform(self, x):
        x = np.array(x)
        assert len(x.shape) == 3
        n1, n2, n_time_points = x.shape

        #-- Get the start-end indices of each window
        min_window_size = self._min_window_size or self._window_size
        window_start = np.array(range(0, n_time_points-min_window_size+1, self._step))
        if len(window_start) == 0:
            #-- There are fewer than window_size time points
            raise Exception('There are only {:} time points, but at least {:} are required for the sliding window'.
                            format(n_time_points, self._min_window_size))
        window_end = window_start + self._window_size
        window_end[-1] = min(window_end[-1], n_time_points)  # make sure that the last window doesn't exceed the input size

        if self._debug:
            win_info = [(s, e, e-s) for s, e in zip(window_start, window_end)]
            print('SlidingWindow transformer: the start,end,length of each sliding window: {:}'.
                  format(win_info))
            if len(win_info) > 1 and win_info[0][2] != win_info[-1][2] and not self._average:
                print('SlidingWindow transformer: note that the last sliding window is smaller than the previous ones, ' +
                      'so the result will be a list of 3-dimensional matrices, with the last list element having ' +
                      'a different dimension than the previous elements. ' +
                      'This format is acceptable by the RiemannDissimilarity transformer')

        if self._average:
            #-- Average the data in each sliding window
            result = np.zeros((n1, n2, len(window_start)))
            for i in range(len(window_start)):
                result[:, :, i] = np.mean(x[:, :, window_start[i]:window_end[i]], axis=2)

        else:
            #-- Don't average the data in each sliding window - just copy it
            result = []
            for i in range(len(window_start)):
                result.append(x[:, :, window_start[i]:window_end[i]])

        return result



def Equalize_event_metadata(epochs, metadataField):
    """
    Take as input an epoch and randomly select the same
    number of trials for each metadataField value
    """

    # extract the metadata Field we want to equilibrate
    metadataname = ['epochs.metadata.{0}'.format(metadataField)];
    meta = eval(metadataname[0]).tolist();
    allConditions = np.unique(meta)  # find the diffent conditions associated with this metadata field
    nBTrialsPerCond = [meta.count(i) for i in allConditions]  # And the number of trials per conditions

    # create a epoch for each condition and select trials
    epochsList = []
    for i in range(len(allConditions)):
        currentepochs = epochs['%s==' % metadataField + '%i' % allConditions[i]]
        Index = list(range(nBTrialsPerCond[i]))
        Index = sorted(rnd.sample(Index, min(nBTrialsPerCond)))
        epochsList.append(currentepochs[Index])

    # concatenate conditions epochs and return
    EqualizedEpochs = mne.concatenate_epochs(epochsList)

    return EqualizedEpochs


