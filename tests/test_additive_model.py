import torch

from tabicl._model.additive import AdditivePerceiver


def _small_model() -> AdditivePerceiver:
    return AdditivePerceiver(
        d_model=16,
        nhead=4,
        col_num_blocks=1,
        col_num_inds=4,
        num_memory_tokens=4,
        num_row_blocks=1,
        num_decoder_blocks=2,
        dim_feedforward=32,
    ).eval()


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
