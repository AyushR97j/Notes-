net = nn.Sequential(
    nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),          # 32x32x16
    nn.MaxPool2d(2),                                    # 16x16x16
    nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),         # 16x16x32
    nn.MaxPool2d(2),                                    # 8x8x32
    nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),  # 4x4x64
    nn.AdaptiveAvgPool2d(1), nn.Flatten(),              # 64
    nn.Linear(64, 10))
