import torch

from vllm._C import graph_ops
torch.manual_seed(0)

N, D_in, H, D_out = 1, 1, 1, 1
model = torch.nn.Sequential(torch.nn.Linear(D_in, H),
                            torch.nn.Dropout(p=0.2),
                            torch.nn.Linear(H, D_out),
                            torch.nn.Dropout(p=0.1)).cuda()
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Placeholders used for capture
static_input = torch.randn(N, D_in, device='cuda')
static_target = torch.randn(N, D_out, device='cuda')

# warmup
# Uses static_input and static_target here for convenience,
# but in a real setting, because the warmup includes optimizer.step()
# you must use a few batches of real data.
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        optimizer.zero_grad(set_to_none=True)
        y_pred = model(static_input)
        loss = loss_fn(y_pred, static_target)
        loss.backward()
        optimizer.step()
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
g.enable_debug_mode()
# Sets grads to None before capture, so backward() will create
# .grad attributes with allocations from the graph's private pool
optimizer.zero_grad(set_to_none=True)
with torch.cuda.graph(g):
    static_y_pred = model(static_input)
    static_loss = loss_fn(static_y_pred, static_target)
    static_loss.backward()
    optimizer.step()

real_inputs = [torch.rand_like(static_input) for _ in range(10)]
real_targets = [torch.rand_like(static_target) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    # Fills the graph's input memory with new data to compute on
    static_input.copy_(data)
    static_target.copy_(target)
    # replay() includes forward, backward, and step.
    # You don't even need to call optimizer.zero_grad() between iterations
    # because the captured backward refills static .grad tensors in place.
    g.replay()
    # Params have been updated. static_y_pred, static_loss, and .grad
    # attributes hold values from computing on this iteration's data.

graph_ops.save_cuda_graph(g)

for data, target in zip(real_inputs, real_targets):
    static_input.copy_(data)
    static_target.copy_(target)
    g.replay()

print(model.state_dict())