"""三个实验共用训练流程；运行方式见 README.md。"""

import argparse
import json

import torch
import torch.nn.functional as F

from model import MiniLanguageModel

VOCAB_SIZE, SEQ_LENGTH, BATCH_SIZE = 10, 8, 32


def make_batch(task, generator):
    if task == "cycle":
        starts = torch.randint(VOCAB_SIZE, (BATCH_SIZE, 1), generator=generator)
        sequence = (starts + torch.arange(SEQ_LENGTH + 1)) % VOCAB_SIZE
        return sequence[:, :-1], sequence[:, 1:]

    inputs = torch.randint(
        VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH), generator=generator,
    )
    targets = torch.full_like(inputs, -100)
    targets[:, 1:] = inputs[:, :-1]
    return inputs, targets


def run(args):
    # 对照实验使用相同初始化、训练数据流和评估数据流。
    torch.manual_seed(args.seed)
    train_rng = torch.Generator().manual_seed(args.seed + 1)
    eval_rng = torch.Generator().manual_seed(args.seed + 2)
    model = MiniLanguageModel(
        VOCAB_SIZE, 32, 4, SEQ_LENGTH, self_only=args.self_only,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    steps = args.steps if args.steps is not None else (300 if args.task == "cycle" else 1000)

    model.train()
    for step in range(steps):
        inputs, targets = make_batch(args.task, train_rng)
        logits = model(inputs)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            print(f"step={step}, loss={loss.item():.4f}")

    # 20 个新 batch；忽略 -100，不将训练 loss 当作评估 loss。
    model.eval()
    correct, count, total_loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in range(20):
            inputs, targets = make_batch(args.task, eval_rng)
            logits = model(inputs)
            predictions = logits.argmax(-1)
            valid = targets != -100
            correct += (predictions[valid] == targets[valid]).sum().item()
            count += valid.sum().item()
            total_loss += F.cross_entropy(
                logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1), reduction="sum",
            ).item()
            if batch == 0:
                print("input:", inputs[0].tolist())
                print("target:", targets[0].tolist())
                print("prediction:", predictions[0].tolist())
                positions = torch.arange(SEQ_LENGTH)
                x = model.token_embedding(inputs) + model.position_embedding(positions)
                _, weights = model.block(x)
                print("attention argmax:", weights[0].argmax(-1).tolist())

    result = dict(
        task=args.task, self_only=args.self_only, seed=args.seed, steps=steps,
        eval_batches=20, evaluated_tokens=count,
        eval_loss=total_loss / count, accuracy=correct / count,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["cycle", "copy"], default="copy")
    parser.add_argument("--self-only", action="store_true", help="只允许关注自己，供消融实验使用")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, help="默认 cycle=300，copy=1000")
    run(parser.parse_args())
