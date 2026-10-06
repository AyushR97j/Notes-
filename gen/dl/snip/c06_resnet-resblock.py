class Block(nn.Module):
    def __init__(self, d, residual):
        super().__init__()
        self.lin, self.bn, self.residual = nn.Linear(d, d), nn.BatchNorm1d(d), residual

    def forward(self, h):
        f = torch.relu(self.bn(self.lin(h)))
        return h + f if self.residual else f      # the only difference: "+ h"
