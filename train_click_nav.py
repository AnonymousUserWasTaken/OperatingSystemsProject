# ============================================================
# FILE: model/showui/train_click_nav.py
# ============================================================

import os
import argparse

import torch
from torch.utils.data import DataLoader

from transformers import (
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)

from .modeling_showui import ShowUIForConditionalGeneration
from .processing_showui import ShowUIProcessor
from .click_nav_dataset import ClickNavVideoDataset, click_nav_collate


def parse_args():
    parser = argparse.ArgumentParser(description="Train ShowUI click navigation with LoRA + 4-bit")

    # --- Core paths ---
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        required=True,
        help="Base pretrained model (e.g. Qwen/Qwen2-VL-2B-Instruct or local ShowUI checkpoint).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Where to save LoRA / fine-tuned weights.",
    )

    # --- Dataset splits ---
    parser.add_argument(
        "--train_split",
        type=str,
        default="train",
        help="Which split directory under OSProject to use for training (default: train).",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        help="Which split under OSProject to use for eval (if exists).",
    )

    # --- Training hyperparams ---
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)

    # --- Precision / quantization ---
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Use fp16 training (if CUDA available).",
    )
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="Load base model in 4-bit (bitsandbytes) to fit in smaller VRAM.",
    )

    # --- LoRA config ---
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of module name substrings to apply LoRA to.",
    )

    # --- Misc ---
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=1000,
        help="Save checkpoint every N steps.",
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=500,
        help="Evaluate every N steps (if eval split exists).",
    )

    args = parser.parse_args()
    return args


def load_processor_and_datasets(args):
    # Processor: same name as base model
    processor = ShowUIProcessor.from_pretrained(args.model_name_or_path)

    train_dataset = ClickNavVideoDataset(split=args.train_split, processor=processor)

    eval_dataset = None
    eval_split_dir = os.path.join(
        os.path.dirname(__file__), "OSProject", args.eval_split
    )
    if os.path.isdir(eval_split_dir):
        eval_dataset = ClickNavVideoDataset(split=args.eval_split, processor=processor)

    return processor, train_dataset, eval_dataset


def load_model_with_lora(args):
    # -------------------------
    # Quantization config (4-bit)
    # -------------------------
    quant_config = None
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )

    # -------------------------
    # Load base model
    # -------------------------
    model = ShowUIForConditionalGeneration.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch_dtype,
        quantization_config=quant_config,
        device_map="auto" if (args.load_in_4bit and torch.cuda.is_available()) else None,
    )

    # We want gradient checkpointing for VRAM savings
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # Prepare for k-bit if quantized
    if args.load_in_4bit:
        model = prepare_model_for_kbit_training(model)

    # -------------------------
    # LoRA config
    # -------------------------
    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    model = get_peft_model(model, lora_config)

    # Helpful debug print to see how many params we're actually training
    model.print_trainable_parameters()

    return model


def main():
    args = parse_args()

    # Set deterministic behavior
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # -----------------------------
    # Load processor + datasets
    # -----------------------------
    processor, train_dataset, eval_dataset = load_processor_and_datasets(args)

    # -----------------------------
    # Load model (4-bit + LoRA)
    # -----------------------------
    model = load_model_with_lora(args)

    # -----------------------------
    # TrainingArguments
    # -----------------------------
    eval_strategy = "no"
    if eval_dataset is not None:
        eval_strategy = "steps"

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy=eval_strategy,
        eval_steps=args.eval_steps,
        save_total_limit=2,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=False,
        report_to=[],  # disable wandb/etc by default
        remove_unused_columns=False,  # VERY IMPORTANT for custom vision inputs
        dataloader_num_workers=2,
        seed=args.seed,
    )

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=click_nav_collate,
    )

    # -----------------------------
    # Train
    # -----------------------------
    trainer.train()

    # -----------------------------
    # Save final adapter + processor
    # -----------------------------
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)

    print(f"Training complete. Saved to {args.output_dir}")


if __name__ == "__main__":
    main()
