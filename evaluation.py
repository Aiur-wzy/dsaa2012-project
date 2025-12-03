import torch
from fer import EmotionCNN, evaluate, build_dataloaders, get_eval_transform


def main():
    # 如果 fer2013.csv 就在 dsaa2012-project-main 目录下，这样写就可以
    # 否则改成绝对/相对正确路径，例如: csv_path = "fer/fer2013.csv"
    csv_path = "fer2013.csv"
    ckpt_path = "runs/exp1/best.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 构建模型并加载权重
    model = EmotionCNN(in_chans=1).to(device)

    # 如果你想去掉 FutureWarning，可以加 weights_only=True（前提是 ckpt 里只保存了 state_dict）
    state = torch.load(ckpt_path, map_location=device)
    # 有的 checkpoint 是 {"state_dict": xxx}，有的是直接保存 state_dict，这里都兼容一下
    state_dict = state["state_dict"] if "state_dict" in state else state
    model.load_state_dict(state_dict)

    # Windows 上为了避免多进程问题，最稳的是先用 num_workers=0
    # 如果后面加速想开多进程，可以把 0 改回 4，前提是 main() 结构保留
    _, _, test_loader = build_dataloaders(
        csv_path,
        batch_size=256,
        num_workers=0,
        in_chans=1,
        eval_transform=get_eval_transform(1),
    )

    criterion = torch.nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print({"test_loss": test_loss, "test_acc": test_acc})


if __name__ == "__main__":
    main()
