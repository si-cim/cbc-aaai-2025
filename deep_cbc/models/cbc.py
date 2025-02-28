"""PyTorch's implementation of the Slim CBC module."""
from typing import Dict, Optional, Tuple, Union

import torch
from torch import Tensor
from torch.nn import Module, Parameter


class SlimClassificationByComponents(Module):
    """
    :param reasoning_labels: Tensor of reasoning labels that define the class mapping
        of the reasoning concepts (one-hot encoded).
    :param init_reasoning_probabilities: The initial tensor used to define the
        trainable parameters ``reasoning_probabilities``. The tensor is assumed to be
        encoded as a breaking chopstick encoding. Consequently, all values must be in
        the unit interval.

    :Shape:

        - ``reasoning_labels``: (*number_of_reasoning_concepts*, *number_of_classes*).
        - ``init_reasoning_probabilities``: (2, *number_of_components*,
          *number_of_reasoning_concepts*).
    """

    def __init__(
        self,
        *,
        reasoning_labels: torch.Tensor,
        init_reasoning_probabilities: Tensor,
        eps: float = 1.0e-7,
    ) -> None:
        r"""Initialize an object of the class."""
        super().__init__()
        self.reasoning_probabilities = Parameter(init_reasoning_probabilities)
        # register input tensor as buffer to ensure movement to correct device
        self.register_buffer("reasoning_labels", reasoning_labels)
        self.eps = eps

        # Note: The PIPNet classification head returns both prediction outputs and pooled distances.
        # So added additional pooled distances return for signature compatability, can be removed as well if needed.
        self.normalization_multiplier = torch.nn.Parameter(
            torch.ones((1,), requires_grad=True)
        )

    @property
    def _decode_reasoning_probabilities(self) -> Tensor:
        r"""Decode the reasoning probabilities.

        The output is a tensor with the positive and negative reasoning probabilities.

        :return: Tensor with the positive and negative reasoning probabilities.

        :Shape: Output: (2, *number_of_components*, *number_of_reasoning_concepts*).
        """
        reasoning_probabilities = torch.softmax(
            self.reasoning_probabilities, dim=0
        ).view(
            2,
            self.reasoning_probabilities.shape[0] // 2,
            self.reasoning_probabilities.shape[1],
        )

        return reasoning_probabilities

    @property
    def effective_reasoning_probabilities(self) -> Tensor:
        r"""Get effective reasoning probabilities.

        The output is a tensor ``p`` with the effect reasoning probabilities, where
        ``p[0]`` is the probability of positive reasoning, ``p[1]`` is the probability
        of indefinite reasoning, and ``p[2]`` is the probability of negative reasoning.
        Note that the sum over the first dimension of the tensor is 1.
        The returned tensor is a detached clone.

        :return: Effective reasoning probabilities.

        :Shape: Output: (3, *number_of_components*, *number_of_reasoning_concepts*).
        """

        return self._decode_reasoning_probabilities.detach().clone()

    def _reasoning(
        self,
        *,
        detection_probabilities: Tensor,
        full_report: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        r"""Wrap the reasoning to realize that the class can be used as a base class.

        :param detection_probabilities: Tensor of probabilities of detected components
            in the inputs.
        :param full_report: If ``False``, only the agreement probability is computed
            (the class probability). If ``True``, all internal probabilities of the
            reasoning process are returned as *detached* and *cloned* tensors. See
            "return" for the returned probabilities.
        :return:
            - If ``full_report==False``, the reasoning probabilities tensor, where
              ``output[i]`` is the probability for reasoning ``i`` if
              ``x.dim()==1``;
            - If ``full_report==False``, the reasoning probabilities tensor, where
              ``output[i,j]`` is the probability for reasoning ``j`` given the input
              ``i`` from the batch of ``detection_probabilities`` if
              ``x.dim()==2``.
            - If ``full_report==True``, for each probability tensor returned in the
              dictionary, the specification for ``full_report==False`` is correct. The
              dictionary holds the following probability tensors: *agreement
              probability* (key 'agreement'), *disagreement probability* (key
              'disagreement'), *detection probability* (key 'detection'; returned for
              completeness), *positive agreement probability* (key 'positive
              agreement'), *negative agreement probability* (key 'negative agreement'),
              *positive disagreement probability* (key 'positive disagreement'),
              *negative disagreement probability* (key 'negative disagreement').

        :Shape:

            - ``detection_probabilities``: (*number_of_components*,) or
              (*batch*, *number_of_components*).
            - Output: If ``full_report==False`` and
              ``detection_probabilities.dim()==1``, (*number_of_reasoning_concepts*,);
              or if ``full_report==False`` and ``detection_probabilities.dim()==2``,
              (*batch*, *number_of_reasoning_concepts*,). If ``full_report==True``,
              dictionary of tensors with the format specified for
              ``full_report==False``.
        """
        reasoning_probabilities = self._decode_reasoning_probabilities

        probabilities = detection_probabilities @ (
            reasoning_probabilities[0] - reasoning_probabilities[1]
        ) + torch.sum(reasoning_probabilities[1], dim=0)

        return probabilities

    def _shared_forward(self, x: Tensor, full_report: bool) -> Tuple[Tensor, Tensor]:
        r"""Shared forward step.

        :param x: Input data tensor.
        :param full_report: If ``False``, only the agreement probability is computed
            (the class probability). If ``True``, all internal probabilities of the
            reasoning process are returned as *detached* and *cloned* tensors. See
            'return' for the returned probabilities.
        :return:
            - If ``full_report==False``, the reasoning probabilities tensor, where
              ``output[i]`` is the probability for reasoning ``i`` if
              ``x.dim()==1``;
            - If ``full_report==False``, the reasoning probabilities tensor, where
              ``output[i,j]`` is the probability for reasoning ``j`` given the input
              ``i`` from the batch of ``detection_probabilities`` if
              ``x.dim()==2``.
            - If ``full_report==True``, for each probability tensor returned in the
              dictionary, the specification for ``full_report==False`` is correct. The
              dictionary holds the following probability tensors: *agreement
              probability* (key 'agreement'), *disagreement probability* (key
              'disagreement'), *detection probability* (key 'detection'; returned for
              completeness), *positive agreement probability* (key 'positive
              agreement'), *negative agreement probability* (key 'negative agreement'),
              *positive disagreement probability* (key 'positive disagreement'),
              *negative disagreement probability* (key 'negative disagreement').

        :Shape: Output: If ``full_report==False`` and
            ``x.dim()==1``,
            (*number_of_reasoning_concepts*,); or if ``full_report==False`` and
            ``x.dim()==2``, (*batch*,
            *number_of_reasoning_concepts*,). If ``full_report==True``, dictionary of
            tensors with the format specified for ``full_report==False``.
        """
        probabilities = self._reasoning(
            detection_probabilities=x,
            full_report=full_report,
        )
        # Note: The PIPNet classification head returns both prediction outputs and pooled distances.
        # So added additional pooled distances return for signature compatability, can be removed as well if needed.
        return (
            probabilities,
            x,
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Compute the class probability (agreement probability) tensor.

        Applies the encoder before the CBC computation.
        Note that we forward the input vector to the ``components`` module in case the
        module expects an input tensor.

        :param x: Input data tensor.
        :return: Tensor of class probabilities.

        :Shape: Output: (*batch*, *number_of_reasoning_concepts*).
        """
        class_probabilities, detection_probabilities = self._shared_forward(
            x, full_report=False
        )
        # Note: The PIPNet classification head returns both prediction outputs and pooled distances.
        # So added additional pooled distances return for signature compatability, can be removed as well if needed.
        return class_probabilities, detection_probabilities

    def all_reasoning_probabilities(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        r"""Compute all internal reasoning probabilities.

        Applies the encoder before the CBC computation.
        Note that we forward the input tensor to the ``components`` module in case the
        module expects an input tensor. All probabilities of the reasoning process are
        returned as *detached* and *cloned* tensors.

        :param x: Input data tensor.
        :return: If ``x.dim()==1`` each tensor in the dict will be one-dimensional,
            where ``output[i]`` is the probability for reasoning ``i``. If
            ``x.dim()==2`` each tensor in the dict will be two-dimensional, where
            ``output[i,j]`` is the probability for reasoning ``j`` given the input
            ``i`` from the batch.
            The dictionary holds the following probability tensors: *agreement
            probability* (key 'agreement'), *disagreement probability* (key
            'disagreement'), *detection probability* (key 'detection'; returned for
            completeness), *positive agreement probability* (key 'positive
            agreement'), *negative agreement probability* (key 'negative agreement'),
            *positive disagreement probability* (key 'positive disagreement'),
            *negative disagreement probability* (key 'negative disagreement').

        :Shape: Output: Dictionary of tensors with the format, if ``x.dim()==1``,
            (*number_of_reasoning_concepts*,); or if ``x.dim()==2``,
            (*batch*, *number_of_reasoning_concepts*,).
        """
        probabilities, detection_probabilities = self._shared_forward(
            x, full_report=True
        )

        return probabilities, detection_probabilities
