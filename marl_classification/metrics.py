from abc import ABC, abstractmethod
from os.path import join
from statistics import mean
from typing import Optional, List, Tuple, Union, Mapping, Any, Generic, TypeVar
import numpy as np

import matplotlib.pyplot as plt
import torch as th
plt.switch_backend('agg')


def format_metric(
		metric: th.Tensor,
		class_map: Mapping[Any, int]
) -> str:

	idx_to_class = {class_map[k]: k for k in class_map}

	return ", ".join(
		[f'\"{idx_to_class[curr_cls]}\" : {metric[curr_cls] * 100.:.1f}%'
		 for curr_cls in range(metric.size()[0])]
	)


T = TypeVar("T")


class Meter(Generic[T], ABC):
	def __init__(self, window_size: Optional[int]) -> None:
		self.__window_size = window_size

		self.__results: List[T] = []

	@abstractmethod
	def _process_value(self, *args) -> T:
		pass

	@property
	def _results(self) -> List[T]:
		return self.__results

	def add(self, *args) -> None:
		if self.__window_size is not None and len(self.__results) >= self.__window_size:
			self.__results.pop(0)

		self.__results.append(self._process_value(*args))

	def set_window_size(self, new_window_size: Union[int, None]) -> None:
		if new_window_size is not None:
			assert new_window_size > 0, f"window size must be > 0 : {new_window_size}"

		self.__window_size = new_window_size


class ConfusionMeter(Meter[Tuple[th.Tensor, th.Tensor]]):
	def __init__(
			self,
			nb_class: int,
			window_size: Optional[int] = None,
	):
		super(ConfusionMeter, self).__init__(window_size)
		self.__nb_class = nb_class

	def _process_value(self, y_proba: th.Tensor, y_true: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
		return y_proba.argmax(dim=1), y_true

	def conf_mat(self) -> th.Tensor:
		y_pred = th.cat([y_p for y_p, _ in self._results], dim=0)
		y_true = th.cat([y_t for _, y_t in self._results], dim=0)

		conf_matrix_indices = th.multiply(y_true, self.__nb_class) + y_pred
		conf_matrix = (
			th.bincount(conf_matrix_indices, minlength=self.__nb_class ** 2)
			.reshape(self.__nb_class, self.__nb_class)
		)

		return conf_matrix

	def accuracy(self) -> th.Tensor:
		conf_mat = self.conf_mat()
		
		diag = th.diagonal(conf_mat, 0)
		sum_diag = diag.sum()
		total = conf_mat.sum()

		accuracy = sum_diag / total
		# print(str(conf_mat))
		# print(str(sum_diag))
		# print(str(total))
		# print(str(accuracy))
		return accuracy

	def precision(self) -> th.Tensor:
		conf_mat = self.conf_mat()

		precs_sum = conf_mat.sum(dim=0) 
		diag = th.diagonal(conf_mat, 0)

		precs = th.zeros(self.__nb_class, device=conf_mat.device)

		mask = precs_sum != 0

		precs[mask] = diag[mask] / precs_sum[mask] 

		return precs

	def recall(self) -> th.Tensor:
		conf_mat = self.conf_mat()

		recs_sum = conf_mat.sum(dim=1)
		diag = th.diagonal(conf_mat, 0)

		recs = th.zeros(self.__nb_class, device=conf_mat.device)

		mask = recs_sum != 0

		recs[mask] = diag[mask] / recs_sum[mask]

		return recs

	def save_conf_matrix_new(
			self,
			epoch: int,
			output_dir: str,
			stage: str
	) -> None:
		# Confusion matrix values
		confusion_matrix = np.array(self.conf_mat().tolist())

		# Define class labels
		class_labels = {
			"benign": 0,
			"malignant": 1
		}

		# Create a heatmap for the confusion matrix
		plt.figure(figsize=(6, 4))
		plt.imshow(confusion_matrix, cmap='summer', interpolation='nearest')

		# Add labels to the tick marks
		plt.xticks(np.arange(len(class_labels)), class_labels.keys())
		plt.yticks(np.arange(len(class_labels)), class_labels.keys())

		# Add text annotations to the cells
		for i in range(len(class_labels)):
			for j in range(len(class_labels)):
				plt.text(j, i, confusion_matrix[i, j], ha='center', va='center', color='black')

		# Add color bar
		plt.colorbar()

		# Set labels for the x and y axes
		plt.xlabel('Predicted')
		plt.ylabel('Actual')

		# Add a title
		plt.title(f"Confusion matrix epoch {epoch} - {stage}")

		plt.savefig(
			join(output_dir, "conf_matrices", f"confusion_matrix_epoch_{epoch}_{stage}.png")
		)

		plt.close()

	def save_conf_matrix(
			self,
			epoch: int,
			output_dir: str,
			stage: str
	) -> None:
		plt.matshow(self.conf_mat().tolist())

		plt.title(f"confusion matrix epoch {epoch} - {stage}")
		plt.ylabel('True Label')
		plt.xlabel('Predicted Label')

		plt.colorbar()

		plt.savefig(
			join(output_dir, "conf_matrices", f"confusion_matrix_epoch_{epoch}_{stage}.png")
		)

		plt.close()


class LossMeter(Meter[float]):
	def __init__(self, window_size: Optional[int]) -> None:
		super(LossMeter, self).__init__(window_size)

	def _process_value(self, value: float) -> float:
		return value

	def loss(self) -> float:
		return mean(self._results)
