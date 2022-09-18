import torch
import torch.nn as nn

# import torch_tensorrt

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True


class Model(nn.Module):

    def forward(self, x):
        x = torch.flip(x, (1,))
        x = torch.cat([x, x], 1)
        return x


model = Model()

# # Compile with Torch TensorRT;
# trt_model = torch_tensorrt.compile(
#     model,
#     inputs=[torch_tensorrt.Input((1, 4))],
#     enabled_precisions={torch.float}  # Run with FP32
# )

model_jit = torch.jit.trace(model, torch.zeros(1, 4))
# print(m)

# Save the model
torch.jit.save(model_jit, "model.pt")
