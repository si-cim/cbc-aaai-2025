"""Module with loss functions."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch
from torch import Tensor
from torch.nn.modules import Module


def _tensor_dimension_and_size_check(
    data_template_comparisons: Tensor,
    template_labels: Tensor,
    data_labels: Tensor,
) -> None:
    r"""Check the sizes and dimensions of the tensors for loss functions.

    The loss is computed based on data-template-comparisons (e.g., prototype to input
    distances or probabilities based on the CBC reasoning), labels of the templates
    (e.g, prototypes), and the data labels. This function checks that the tensors have
    the correct sizes.

    Note that the function does not check the one-hot coding of the labels.

    :param data_template_comparisons: Tensor of distances, probabilities, etc. of size
        (number_of_samples, number_of_prototypes).
    :param template_labels: The labels of the templates (e.g., reasoning plans or
        prototypes) one-hot coded. The size must be
        (number_of_prototypes, number_of_classes).
    :param data_labels: The labels of the data one-hot coded. The size must be
        (number_of_samples, number_of_classes).
    :raises ValueError: If tensors are not of class:`Tensor`.
    :raises ValueError: If tensors are not of dimension 2.
    :raises ValueError: If the tensor sizes do not match as specified.
    """
    if (
        not isinstance(data_template_comparisons, Tensor)
        or not isinstance(template_labels, Tensor)
        or not isinstance(data_labels, Tensor)
    ):
        raise ValueError(
            f"The inputs must be of class Tensor. "
            f"Provided type(data_template_comparisons)="
            f"{type(data_template_comparisons)}, "
            f"type(template_labels)={type(template_labels)}, and "
            f"type(data_labels)={type(data_labels)}."
        )

    if (
        data_template_comparisons.dim() != 2
        or template_labels.dim() != 2
        or data_labels.dim() != 2
    ):
        raise ValueError(
            f"The dimensions of the tensors must be 2. "
            f"Provided data_template_comparisons.dim()="
            f"{data_template_comparisons.dim()}, "
            f"template_labels.dim()={template_labels.dim()} (one-hot coding), "
            f"and data_labels.dim()={data_labels.dim()} (one-hot coding)."
        )

    if data_template_comparisons.size()[1] != template_labels.size()[0]:
        raise ValueError(
            f"The number of templates does not match: "
            f"data_template_comparisons.size()[1]="
            f"{data_template_comparisons.size()[1]} != "
            f"template_labels.size()[0]={template_labels.size()[0]}."
        )

    if data_template_comparisons.size()[0] != data_labels.size()[0]:
        raise ValueError(
            f"The number of samples does not match: "
            f"data_template_comparisons.size()[0]="
            f"{data_template_comparisons.size()[0]} != "
            f"data_labels.size()[0]={data_labels.size()[0]}."
        )

    if data_labels.size()[1] != template_labels.size()[1]:
        raise ValueError(
            f"The number of classes does not match (one-hot coding): "
            f"data_labels.size()[1]={data_labels.size()[1]} != "
            f"template_labels.size()[1]={template_labels.size()[1]}."
        )


def _closest_correct_and_incorrect_distance(
    *,
    distances: Tensor,
    prototype_labels: Tensor,
    data_labels: Tensor,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    r"""Determine the closest correct and incorrect prototype.

    :param distances: Tensor of distances.
    :param prototype_labels: The labels of the prototypes one-hot encoded.
    :param data_labels: The labels of the data one-hot encoded.
    :return: Tuple of tensors of (``distance_closest_correct``,
        ``idx_closest_correct``, ``distance_closest_incorrect``,
        ``idx_closest_incorrect``).

    :Shape:

        - ``distances``: (*number_of_samples*, *number_of_prototypes*).
        - ``prototype_labels``: (*number_of_prototypes*, *number_of_classes*).
        - ``data_labels``: (*number_of_samples*, *number_of_classes*).
        - Output: Each tensor has the following shape (*number_of_samples*,).
    """
    # compute a value that is larger than any distance
    max_distances = torch.max(distances) + 1

    # compute a matrix where one means that this data sample has the same label as the
    # prototype (number_samples, number_prototypes)
    labels_agree = data_labels @ prototype_labels.T

    # increase distance by max_distances for all prototypes that are not correct
    distance_closest_correct, idx_closest_correct = torch.min(
        distances + (1 - labels_agree) * max_distances, 1
    )

    distance_closest_incorrect, idx_closest_incorrect = torch.min(
        distances + labels_agree * max_distances, 1
    )

    return (
        distance_closest_correct,
        idx_closest_correct,
        distance_closest_incorrect,
        idx_closest_incorrect,
    )


def margin_loss(
    *,
    data_template_comparisons: Tensor,
    template_labels: Tensor,
    data_labels: Tensor,
    margin: float,
    similarity: bool,
) -> Tensor:
    r"""Margin loss function.

    Functional implementation of the :class:`.MarginLoss`. See this class for
    further information.

    :param data_template_comparisons: Tensor of comparison values. This could be a
        vector of distances or probabilities that is later used to determine the
        output class.
    :param template_labels: The labels of the templates one-hot encoded.
    :param data_labels: The labels of the data one-hot encoded.
    :param margin: Positive value that specifies the minimal margin to be achieved
        between the best correct and incorrect template.
    :param similarity: A binary parameter that determines whether similarities
        (maximum determines the winner) or dissimilarities (minimum determines the
        winner) are expected for ``data_template_comparisons``.
    :return: Tensor of loss values for each sample.

    :Shape:

        - ``data_template_comparisons``: (*number_of_samples*, *number_of_templates*).
        - ``template_labels``: (*number_of_templates*, *number_of_classes*).
        - ``data_labels``: (*number_of_samples*, *number_of_classes*).
        - Output: (*number_of_samples*,).

    :Example:

    >>> data_template_comparisons = torch.rand(64, 4)
    >>> template_labels = torch.concatenate([torch.eye(2), torch.eye(2)])
    >>> class_labels = (torch.randn(64)<0).float()
    >>> data_labels = torch.vstack([1 - class_labels, class_labels]).T
    >>> output = loss.margin_loss(
    ...     data_template_comparisons=data_template_comparisons,
    ...     template_labels=template_labels,
    ...     data_labels=data_labels,
    ...     margin=0.3,
    ...     similarity=True
    ... )
    """
    _tensor_dimension_and_size_check(
        data_template_comparisons=data_template_comparisons,
        template_labels=template_labels,
        data_labels=data_labels,
    )

    if similarity:
        if template_labels.size()[0] == template_labels.size()[1]:
            max_probability_per_class = data_template_comparisons @ template_labels
        else:
            max_probability_per_class, _ = torch.max(
                template_labels.unsqueeze(0) * data_template_comparisons.unsqueeze(-1),
                dim=1,
            )

        # In the following, the minus is required to have a minimization problem:
        # best_correct -> highest_correct_probability
        best_correct = -torch.sum(data_labels * max_probability_per_class, dim=1)

        # best_incorrect -> highest_incorrect_probability
        best_incorrect, _ = torch.max(max_probability_per_class - data_labels, dim=1)
        best_incorrect = -best_incorrect

    else:
        # best_correct -> distance_closest_correct
        # best_incorrect -> distance_closest_incorrect
        (best_correct, _, best_incorrect, _) = _closest_correct_and_incorrect_distance(
            distances=data_template_comparisons,
            prototype_labels=template_labels,
            data_labels=data_labels,
        )

    loss_per_sample = torch.relu(best_correct - best_incorrect + margin)

    return loss_per_sample


class TemplateInputComparisonBasedLoss(Module, ABC):
    r"""Base class for a template-input-comparison-based loss.

    Can be used to implement custom loss functions. See the classes :class:`.GLVQLoss`
    or :class:`.MarginLoss` for examples. The naming is generic so that 'comparison'
    could mean distances or similarities and 'templates' could be prototypes or
    components.

    :param dimension_and_size_check: Determines whether the dimensions and sizes of the
        inputs are checked. If this check is already done in the loss function call, set
        this to False.
    """

    def __init__(self, dimension_and_size_check: bool = True):
        r"""Initialize an object of the class."""
        super().__init__()
        self.dimension_and_size_check = dimension_and_size_check

    def forward(
        self,
        data_template_comparisons: Tensor,
        template_labels: Tensor,
        data_labels: Tensor,
    ) -> Tensor:
        r"""Forward pass.

        Calls the :meth:`.loss_function` and depending on ``dimension_and_size_check``
        the dimension and size check.

        :param data_template_comparisons: Tensor of distances or similarities.
        :param template_labels: The labels of the templates one-hot encoded.
        :param data_labels: The labels of the data one-hot encoded.
        :return: Tensor of loss values for each sample.

        :Shape:

            - ``data_template_comparisons``:
              (*number_of_samples*, *number_of_templates*).
            - ``template_labels``: (*number_of_templates*, *number_of_classes*).
            - ``data_labels``: (*number_of_samples*, *number_of_classes*).
            - Output: (*number_of_samples*,).
        """
        if self.dimension_and_size_check:
            _tensor_dimension_and_size_check(
                data_template_comparisons=data_template_comparisons,
                template_labels=template_labels,
                data_labels=data_labels,
            )

        loss = self.loss_function(
            data_template_comparisons,
            template_labels,
            data_labels,
        )

        return loss

    @abstractmethod
    def loss_function(
        self,
        data_template_comparisons: Tensor,
        template_labels: Tensor,
        data_labels: Tensor,
    ) -> Tensor:
        r"""Abstract method for the loss function implementation.

        The loss function should be implemented here. To support different namings of
        the arguments this function is called by :meth:`.forward` without keywords.
        Therefore, **do not** change the order of the arguments when implementing this
        method.

        :param data_template_comparisons: Tensor of distances or similarities.
        :param template_labels: The labels of the templates one-hot encoded.
        :param data_labels: The labels of the data one-hot encoded.
        :return: Tensor of loss values for each sample.

        :Shape:

            - ``data_template_comparisons``:
              (*number_of_samples*, *number_of_templates*).
            - ``template_labels``: (*number_of_templates*, *number_of_classes*).
            - ``data_labels``: (*number_of_samples*, *number_of_classes*).
        """


class MarginLoss(TemplateInputComparisonBasedLoss):
    r"""Computes the Margin loss for similarities or dissimilarities.

    This loss is the margin loss implementation with a specifiable ``margin``
    value. If ``similarity`` is ``True``, then the loss assumes that the comparison
    values are based on similarities (e.g., probabilities) so that large values mean
    high similarity. In this case, the loss is given by

    .. math::
        \mathrm{loss}(\mathrm{all\_templates},x) =
        \max\left\{ s^- - s^+ + \mathrm{margin}, 0 \right\},

    where :math:`s^+` is the highest similarity of a template of the same class as the
    input and :math:`s^-` is the highest similarity of a template of a different
    class than the input :math:`x`. In case of, dissimilarities (e.g., distance
    functions) the loss becomes

    .. math::
        \mathrm{loss}(\mathrm{all\_templates},x) =
        \max\left\{ d^+ - d^- + \mathrm{margin}, 0 \right\},

    where :math:`d^+` is the smallest dissimilarity of a template of the same class as
    the input and :math:`d^-` is the smallest dissimilarity of a template of a
    different class than the input :math:`x`.

    The function uses the word "template" instead of using the names "prototypes" or
    "components". However, depending on the use case, a "template" could be a
    "prototype". Moreover, instead of "similarity" or "dissimilarity", the function
    uses the word 'comparison'.

    Note that the loss values are always returned **not accumulated**. This means the
    loss value for each sample is returned.

    :param margin: The margin value of the loss function. Usually, a positive value.
    :param similarity: A boolean value to specify if the input is a similarity or
        dissimilarity.

    :Example:

    >>> distances = torch.rand(64, 4)
    >>> prototype_labels = torch.concatenate([torch.eye(2), torch.eye(2)])
    >>> class_labels = (torch.randn(64)<0).float()
    >>> data_labels = torch.vstack([1 - class_labels, class_labels]).T
    >>> loss_func = loss.MarginLoss(margin=0.3, similarity=False)
    >>> output = loss_func(distances, prototype_labels, data_labels)
    """

    def __init__(self, margin: float, similarity: bool) -> None:
        r"""Initialize an object of the class."""
        super().__init__(dimension_and_size_check=False)
        self.margin = margin
        self.similarity = similarity

    def loss_function(
        self,
        data_template_comparisons: Tensor,
        template_labels: Tensor,
        data_labels: Tensor,
    ) -> Tensor:
        r"""Margin loss computation.

        :param data_template_comparisons: Tensor of comparison values. This could be a
            vector of distances or probabilities that is later used to determine the
            output class.
        :param template_labels: The labels of the templates one-hot encoded.
        :param data_labels: The labels of the data one-hot encoded.
        :return: Tensor of loss values for each sample.

        :Shape:

            - ``data_template_comparisons``: (*number_of_samples*,
              *number_of_templates*).
            - ``template_labels``: (*number_of_templates*, *number_of_classes*).
            - ``data_labels``: (*number_of_samples*, *number_of_classes*).
            - Output: (*number_of_samples*,).

        :Example:
        """
        loss = margin_loss(
            data_template_comparisons=data_template_comparisons,
            template_labels=template_labels,
            data_labels=data_labels,
            margin=self.margin,
            similarity=self.similarity,
        )

        return loss


''' The following function is not supported and  is in experimental mode. The function
could be supported in later releases if we see that there is a need. In this case, we
try to support the automatic computation of the dual norm. (Sascha, 05.02.2024)

def improved_robust_glvq_loss(
    *,
    distances: Tensor,
    prototype_labels: Tensor,
    data_labels: Tensor,
    prototype_distances: Tensor,
) -> Tensor:
    """ Loss according to the paper of Hein.

    Currently, fixed to the L2 distance because the normalizer must be the dual norm.

    :param distances:
    :param prototype_labels:
    :param data_labels:
    :param prototype_distances:
    :return:
    """
    _tensor_dimension_and_size_check(
        data_template_comparisons=distances,
        template_labels=prototype_labels,
        data_labels=data_labels,
    )

    (
        distance_closest_correct,
        idx_closest_correct,
        distance_closest_incorrect,
        idx_closest_incorrect,
    ) = _closest_correct_and_incorrect_distance(
        distances=distances, prototype_labels=prototype_labels, data_labels=data_labels
    )

    normalizer = prototype_distances[idx_closest_correct, idx_closest_incorrect]

    loss_per_sample = (
        (distance_closest_correct**2 - distance_closest_incorrect**2)
        / 2
        * normalizer
    )

    return loss_per_sample
'''
