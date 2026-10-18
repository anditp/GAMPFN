import pytest
import torch
from torch.autograd.graph import saved_tensors_hooks

from tabicl._model.additive import AdditivePerceiver


def _small_model(**kwargs) -> AdditivePerceiver:
    kwargs.setdefault("row_num_inds", 4)
    kwargs.setdefault("decoder_chunk_size", 2)
    return AdditivePerceiver(
        d_model=16,
        nhead=4,
        col_num_blocks=1,
        col_num_inds=4,
        num_memory_tokens=4,
        num_row_blocks=1,
        num_decoder_blocks=2,
        dim_feedforward=32,
        **kwargs,
    ).eval()


def _saved_activation_bytes_per_sample(model: AdditivePerceiver, batch_size: int) -> int:
    parameter_storages = {parameter.untyped_storage().data_ptr() for parameter in model.parameters()}
    saved_storages = {}

    def pack(tensor):
        storage = tensor.untyped_storage()
        pointer = storage.data_ptr()
        if pointer not in parameter_storages:
            saved_storages[pointer] = storage.nbytes()
        return tensor

    model.train()
    x_context = torch.randn(batch_size, 12, 4)
    y_context = torch.randn(batch_size, 12)
    x_query = torch.randn(batch_size, 12, 4)
    with saved_tensors_hooks(pack, lambda tensor: tensor):
        prediction, effects = model(x_context, y_context, x_query)
        (prediction.square().mean() + effects.square().mean()).backward()
    return sum(saved_storages.values()) // batch_size


def test_additive_prediction_is_centered_and_exactly_decomposed():
    torch.manual_seed(0)
    model = _small_model()
    x_context = torch.randn(2, 7, 3)
    y_context = torch.randn(2, 7)
    x_query = torch.randn(2, 5, 3)

    column_shapes, row_shapes = [], []
    hook = model.col_embedder.register_forward_hook(
        lambda _module, _inputs, output: column_shapes.append(output.shape)
    )
    row_hook = model.row_contextualizer[0].register_forward_hook(
        lambda _module, _inputs, output: row_shapes.append(output.shape)
    )
    context = model.encode_context(x_context, y_context)
    hook.remove()
    row_hook.remove()
    context_effects = model.predict_effects(context, x_context)
    prediction, effects = model(x_context, y_context, x_query)

    assert column_shapes == [torch.Size((2, 7, 3, 16))]
    assert row_shapes == [torch.Size((2, 7, 3, 16))]
    assert context.memory.shape == (2, 3, 4, 16)
    assert prediction.shape == (2, 5)
    assert effects.shape == (2, 5, 3)
    assert torch.allclose(context_effects.mean(dim=1), torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(prediction, context.y_mean + effects.sum(dim=-1), atol=1e-6)


def test_probabilistic_prediction_has_centered_quantile_effects_and_exact_decomposition():
    torch.manual_seed(9)
    model = _small_model(num_quantiles=5)
    x_context = torch.randn(2, 7, 3)
    y_context = torch.randn(2, 7)
    x_query = torch.randn(2, 5, 3)

    context = model.encode_context(x_context, y_context)
    context_effects = model.predict_effects(context, x_context)
    prediction, effects = model(x_context, y_context, x_query)

    assert context.quantile_intercept.shape == (2, 1, 5)
    assert prediction.shape == (2, 5, 5)
    assert effects.shape == (2, 5, 3, 5)
    assert torch.allclose(
        context_effects.mean(dim=1),
        torch.zeros(2, 3, 5),
        atol=1e-6,
    )
    assert torch.allclose(
        prediction,
        context.quantile_intercept + effects.sum(dim=2),
        atol=1e-6,
    )


def test_probabilistic_prediction_rejects_nonadditive_residual():
    with pytest.raises(ValueError, match="do not support an interaction"):
        _small_model(num_quantiles=5, interaction_residual=True)


def test_interaction_residual_starts_at_zero_and_is_context_centered():
    torch.manual_seed(4)
    model = _small_model(interaction_residual=True)
    x_context = torch.randn(2, 7, 4)
    y_context = torch.randn(2, 7)
    x_query = torch.randn(2, 5, 4)

    prediction, effects, interaction = model(
        x_context, y_context, x_query, return_interaction=True
    )

    assert torch.count_nonzero(interaction) == 0
    assert torch.allclose(prediction, y_context.mean(dim=1, keepdim=True) + effects.sum(dim=-1))

    with torch.no_grad():
        model.interaction_head.weight.fill_(0.1)
    context = model.encode_context(x_context, y_context)
    context_interaction = model.predict_interaction(context, x_context)
    assert torch.allclose(
        context_interaction.mean(dim=1), torch.zeros(2), atol=1e-6
    )


def test_interaction_residual_is_feature_permutation_invariant():
    torch.manual_seed(6)
    model = _small_model(interaction_residual=True)
    with torch.no_grad():
        model.interaction_head.weight.normal_()
    x_context = torch.randn(2, 8, 4)
    y_context = torch.randn(2, 8)
    x_query = torch.randn(2, 5, 4)
    feature_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    permutation = torch.tensor([2, 0, 3, 1])

    prediction, effects, interaction = model(
        x_context,
        y_context,
        x_query,
        feature_mask,
        return_interaction=True,
    )
    permuted_prediction, permuted_effects, permuted_interaction = model(
        x_context[:, :, permutation],
        y_context,
        x_query[:, :, permutation],
        feature_mask[:, permutation],
        return_interaction=True,
    )

    assert torch.allclose(permuted_prediction, prediction, atol=1e-5)
    assert torch.allclose(permuted_effects, effects[:, :, permutation], atol=1e-5)
    assert torch.allclose(permuted_interaction, interaction, atol=1e-5)


def test_effect_queries_do_not_depend_on_the_other_query_values():
    torch.manual_seed(1)
    model = _small_model()
    x_context = torch.randn(2, 8, 4)
    y_context = torch.randn(2, 8)
    x_query = torch.randn(2, 6, 4)
    context = model.encode_context(x_context, y_context)

    effects_together = model.predict_effects(context, x_query)
    effects_alone = model.predict_effects(context, x_query[:, :1])

    assert torch.allclose(effects_together[:, :1], effects_alone, atol=1e-6)


def test_feature_permutation_equivariance_masking_and_gradients():
    torch.manual_seed(2)
    model = _small_model().train()
    x_context = torch.randn(2, 6, 4)
    y_context = torch.randn(2, 6)
    x_query = torch.randn(2, 3, 4)
    feature_mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    permutation = torch.tensor([2, 0, 3, 1])

    prediction, effects = model(x_context, y_context, x_query, feature_mask)
    permuted_prediction, permuted_effects = model(
        x_context[:, :, permutation],
        y_context,
        x_query[:, :, permutation],
        feature_mask[:, permutation],
    )

    assert torch.count_nonzero(effects[0, :, 3]) == 0
    assert torch.allclose(permuted_prediction, prediction, atol=1e-5)
    assert torch.allclose(permuted_effects, effects[:, :, permutation], atol=1e-5)

    (prediction.square().mean() + effects.square().mean()).backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_recompute_and_chunking_reduce_saved_backward_memory_per_sample():
    torch.manual_seed(3)
    unchunked = _small_model(decoder_chunk_size=None, recompute=False)
    scalable = _small_model(decoder_chunk_size=2, recompute=True)
    scalable.load_state_dict(unchunked.state_dict())

    unchunked_bytes = _saved_activation_bytes_per_sample(unchunked, batch_size=2)
    scalable_bytes = _saved_activation_bytes_per_sample(scalable, batch_size=2)

    assert scalable_bytes < unchunked_bytes, (
        f"expected scalable backward memory per sample ({scalable_bytes} bytes) "
        f"to be below unchunked memory ({unchunked_bytes} bytes)"
    )
