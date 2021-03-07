# The LRP model needs help from torch
import torch
import torch.nn as nn

# import base LRP model
from .BaseLRP import BaseLRP


class ConvLRP(BaseLRP):
    def __init__(self, conv_model):
        self.model = conv_model

    def explain(self, input, epsilon=0.005):
        return self._explain_convolution(input, epsilon=epsilon)

    def _explain_convolution(self, input, epsilon=0.005) -> torch.Tensor:
        """Because using `_layerwise_relevance_propagation`, the algorithm uses the following steps.
        1. Convert input by `_image_to_column` to calculation easier, say input_conv
        2. Calculate LRP by input_conv and output.
        3. Restore start_layer_relevance by `_column_to_image`, say start_layer_relevance_restore. This is the last start_layer.
        """
        model_info = self._derive_conv_info(self.model)

        input_conv = self._image_to_column(
            input,
            kernel_h=model_info["kernel_h"],
            kernel_w=model_info["kernel_w"],
            padding=model_info["padding"],
            stride=model_info["stride"],
        )

        start_layer_relevance = self._layerwise_relevance_propagation(
            self.model, input_conv, self.model(input)
        )

        start_layer_relevance = self._column_to_image(
            start_layer_relevance,
            input.size(),
            kernel_h=model_info["kernel_h"],
            kernel_w=model_info["kernel_w"],
            padding=model_info["padding"],
            stride=model_info["stride"],
        )

        return start_layer_relevance

    def forward(self, input):
        model_info = self._derive_conv_info(self.model)

        n, c, h, w = input.size()

        input, out_h, out_w = self._image_to_column(
            input,
            kernel_h=model_info["kernel_h"],
            kernel_w=model_info["kernel_w"],
            padding=model_info["padding"],
            stride=model_info["stride"],
        )

        weight = self.model.weight
        weight = weight.reshape(weight.size()[0], -1).transpose(0, 1)

        bias = self.model.bias
        print(bias.size())

        return (input @ weight).reshape(n, out_h, out_w, -1).permute(0, 3, 1, 2)

    def _image_to_column(
        self,
        input_data: torch.Tensor,
        kernel_h: int,
        kernel_w: int,
        stride: int = 1,
        padding=0,
    ) -> torch.Tensor:
        """Implement traditional img2col function.

        The function makes calculation easier and faster even the space complexity is sacrificed.

        Parameter
        =========
        input_data: input data
        kernel_h: The height of kernel
        kernel_w: The width of kernel

        Algorithm Description
        =====================
        1. Initialize output data with expected height and width for output data
            + Derive `output_h` and `output_w`
            + Initialize `output` by `torch.zeros`
        2. Pad on the input, called `padded_input`.
        3. Iterating with the kernel size, save kernels' treatment area.
        4. Resizing the output data.
        """
        batch, channel, height, width = input_data.size()

        output_h = (height + 2 * padding - kernel_h) // stride + 1
        output_w = (width + 2 * padding - kernel_w) // stride + 1

        output = torch.zeros(batch, channel, kernel_h, kernel_w, output_h, output_w)

        padded_input = nn.functional.pad(
            input_data, [padding, padding, padding, padding]
        )

        for y in range(kernel_h):
            for x in range(kernel_w):
                output[:, :, y, x, :, :] = padded_input[
                    :,
                    :,
                    y : y + stride * output_h : stride,
                    x : x + stride * output_w : stride,
                ]

        return (
            output.permute(0, 4, 5, 1, 2, 3).reshape(batch * output_h * output_w, -1),
            output_h,
            output_w,
        )

    def _column_to_image(
        self,
        output: torch.Tensor,
        input_size: torch.Size,
        kernel_h: int,
        kernel_w: int,
        stride: int = 1,
        padding: int = 0,
    ) -> torch.Tensor:
        batch, channel, height, width = input_size
        output_h = (height + 2 * padding - kernel_h) // stride + 1
        output_w = (width + 2 * padding - kernel_w) // stride + 1

        output = output.reshape(
            batch, output_h, output_w, channel, kernel_h, kernel_w
        ).transpose(0, 3, 4, 5, 1, 2)

        input = torch.zeros(
            batch,
            channel,
            height + 2 * padding + stride - 1,
            width + 2 * padding + stride - 1,
        )

        for y in range(kernel_h):
            for x in range(kernel_w):
                input[
                    :,
                    :,
                    y : y + stride * output_h : stride,
                    x : x + stride * output_w : stride,
                ] += output[:, :, y, x, :, :]

        return input[:, :, padding : height + padding, padding : width + padding]

    def _derive_conv_info(self, model):
        return {
            "kernel_h": model.kernel_size[0],
            "kernel_w": model.kernel_size[1],
            "padding": model.padding[0],
            "stride": model.stride[0],
        }